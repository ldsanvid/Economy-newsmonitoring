from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import re
from datetime import datetime, timedelta
import dateparser
from dateparser.search import search_dates
import csv
from wordcloud import WordCloud
from flask import send_file
from collections import OrderedDict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from email.utils import formataddr
from dotenv import load_dotenv


from babel.dates import format_date
from s3_utils import s3_download_all as r2_download_all, s3_upload as r2_upload
import numpy as np
import faiss

# LangChain / RAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def nombre_mes(fecha):
    """Devuelve la fecha con mes en español, ej: 'agosto 2025'"""
    return format_date(fecha, "LLLL yyyy", locale="es").capitalize()


# ------------------------------
# 🔑 Configuración API y Flask
# ------------------------------
load_dotenv()
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# 🔄 Sincronizar índices y metadatos desde Cloudflare R2 al iniciar
try:
    r2_download_all()
    print("✅ Archivos FAISS/CSV sincronizados desde R2")
except Exception as e:
    print(f"⚠️ No se pudo sincronizar desde R2: {e}")

@app.route("/")
def home():
    return send_file("index.html")
# ------------------------------
# 📂 Carga única de datos — con rutas absolutas seguras
base_dir = os.path.dirname(os.path.abspath(__file__))

print("📁 Base directory:", base_dir)

# --- Cargar base de noticias ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    noticias_path = os.path.join(base_dir, "noticias_fondo_fuentes_rango_03-07-2025.csv")
    print("📁 Base directory:", base_dir)
    print("Intentando leer:", noticias_path)

    df = pd.read_csv(noticias_path, encoding="utf-8")
    print(f"✅ Noticias cargadas: {len(df)} filas")
    print("🧩 Columnas detectadas:", list(df.columns))

    # Detectar automáticamente la columna de fecha
    fecha_col = next((c for c in df.columns if "fecha" in c.lower()), None)
    if fecha_col:
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce", dayfirst=True)
        df = df.rename(columns={fecha_col: "Fecha"}).dropna(subset=["Fecha"])
        print(f"📅 Columna '{fecha_col}' convertida correctamente. Rango:",
              df["Fecha"].min(), "→", df["Fecha"].max())
    else:
        print("⚠️ No se encontró columna con 'fecha' en el nombre.")
        df["Fecha"] = pd.NaT
# 🔗 ---------- LangChain: embeddings, vectorstores y LLM ----------

    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
)

    vectorstore_noticias = None
    retriever_noticias = None

    vectorstore_resumenes = None
    retriever_resumenes = None

    # ------------------------------
    # 🔗 MODELO LLM Y CHAIN PARA /pregunta
    # ------------------------------

    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key,
    )

    prompt_pregunta = ChatPromptTemplate.from_messages([
        ("system", """
Eres un analista experto en noticias, geopolítica y economía.
Responde SIEMPRE en español.
NO inventes datos ni traigas información de fuera del contexto.
Si el contexto incluye al menos un titular o un resumen relevante, NO digas que “no se dispone de información” ni frases parecidas; en su lugar, explica lo que SÍ se sabe con base en esos elementos.
Solo si el contexto está totalmente vacío (sin titulares ni resúmenes sobre el tema) puedes decir que no hay información disponible.
Tu objetivo es responder la pregunta del usuario de forma profesional, clara y basada en los titulares y resúmenes proporcionados.
"""),
        ("user", "{texto_usuario}")
    ])


    chain_pregunta = prompt_pregunta | llm_chat | StrOutputParser()
    

    def cargar_vectorstore_noticias(df_noticias: pd.DataFrame):
        """
        Construye o actualiza de forma incremental el vectorstore de noticias.

        - Primera vez: embebe todas las noticias y crea el índice.
        - Siguientes veces: detecta qué filas del df no están todavía embebidas
        (por clave única) y solo calcula embeddings para esas noticias nuevas.
    """
        global vectorstore_noticias, retriever_noticias

        if df_noticias is None or df_noticias.empty:
            print("⚠️ df_noticias vacío, no se construye vectorstore_noticias")
            vectorstore_noticias = None
            retriever_noticias = None
            return

        # 📁 Directorio base para guardar índice y metadatos de LangChain
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_dir = os.path.join(base_dir, "faiss_index", "noticias_lc")
        os.makedirs(index_dir, exist_ok=True)
        meta_path = os.path.join(index_dir, "noticias_lc_metadata.csv")

        # 1️⃣ Construir clave única para cada noticia del df actual
        df_noticias = df_noticias.copy()

        def make_unique_key(row):
            titulo = str(row.get("Título", "")).strip()
            fuente = str(row.get("Fuente", "")).strip()
            fecha_val = row.get("Fecha", None)
            if pd.notnull(fecha_val):
                try:
                    fecha_iso = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                except Exception:
                    fecha_iso = ""
            else:
                fecha_iso = ""
            return f"{fecha_iso}|{fuente}|{titulo}"

        df_noticias["unique_key_lc"] = df_noticias.apply(make_unique_key, axis=1)

        # 2️⃣ Leer metadatos previos (si existen) para saber qué noticias ya tienen embedding
        existing_keys = set()
        df_meta_prev = None
        if os.path.exists(meta_path):
            try:
                df_meta_prev = pd.read_csv(meta_path, encoding="utf-8")
                if "unique_key_lc" in df_meta_prev.columns:
                    existing_keys = set(df_meta_prev["unique_key_lc"].astype(str))
                print(f"ℹ️ Metadatos previos cargados: {len(existing_keys)} noticias embebidas.")
            except Exception as e:
                print(f"⚠️ Error al leer metadatos previos de noticias: {e}")
                df_meta_prev = None
                existing_keys = set()

        # 3️⃣ Detectar noticias nuevas (filas cuyo unique_key_lc no está en existing_keys)
        mask_new = ~df_noticias["unique_key_lc"].isin(existing_keys)
        df_new = df_noticias[mask_new].copy()

        if df_meta_prev is None:
            df_meta_prev = pd.DataFrame(columns=[
                "unique_key_lc", "Fecha", "Título", "Fuente",
                "Enlace", "Cobertura", "Término", "Sentimiento", "Idioma"
            ])

        # 4️⃣ Cargar índice previo de LangChain (si existe)
        vectorstore_noticias = None
        if os.path.isdir(index_dir) and any(f.endswith(".faiss") for f in os.listdir(index_dir)):
            try:
                vectorstore_noticias = LCFAISS.load_local(
                    index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ vectorstore_noticias existente cargado desde disco.")
            except Exception as e:
                print(f"⚠️ No se pudo cargar vectorstore_noticias existente, se reconstruirá desde cero: {e}")
                vectorstore_noticias = None

        # 5️⃣ Construir Document para noticias nuevas
        docs_nuevos = []
        for _, row in df_new.iterrows():
            titulo = str(row.get("Título", "")).strip()
            if not titulo:
                continue

            fecha_val = row.get("Fecha", None)
            if pd.notnull(fecha_val):
                try:
                    fecha_str = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                except Exception:
                    fecha_str = None
            else:
                fecha_str = None

            metadata = {
                "fecha": fecha_str,
                "fuente": row.get("Fuente"),
                "enlace": row.get("Enlace"),
                "cobertura": row.get("Cobertura"),
                "sentimiento": row.get("Sentimiento"),
                "termino": row.get("Término"),
                "idioma": row.get("Idioma"),
                "unique_key_lc": row.get("unique_key_lc"),
            }

            docs_nuevos.append(Document(page_content=titulo, metadata=metadata))

        # 6️⃣ Crear o actualizar el vectorstore de noticias
        if vectorstore_noticias is None:
            # Primera vez: si no hay índice previo, construirlo desde cero con TODO lo nuevo
            if docs_nuevos:
                print(f"🧩 Construyendo vectorstore_noticias desde cero con {len(docs_nuevos)} noticias…")
                vectorstore_noticias = LCFAISS.from_documents(docs_nuevos, embeddings)
            else:
                print("⚠️ No hay documentos nuevos y no existe índice previo; no se construye vectorstore_noticias.")
                retriever_noticias = None
                return
        else:
            # Ya había índice previo: solo agregamos los documentos nuevos
            if docs_nuevos:
                print(f"🧩 Agregando {len(docs_nuevos)} noticias nuevas a vectorstore_noticias…")
                vectorstore_noticias.add_documents(docs_nuevos)
            else:
                print("ℹ️ No hay noticias nuevas para agregar. Se usa el índice existente.")

        # 7️⃣ Actualizar metadatos y guardar
        if not df_new.empty:
            df_meta_new = df_new[[
                "unique_key_lc", "Fecha", "Título", "Fuente",
                "Enlace", "Cobertura", "Término", "Sentimiento", "Idioma"
            ]].copy()
            df_meta_final = pd.concat([df_meta_prev, df_meta_new], ignore_index=True)
        else:
            df_meta_final = df_meta_prev

        try:
            df_meta_final.to_csv(meta_path, index=False, encoding="utf-8-sig")
            print(f"✅ Metadatos de noticias guardados/actualizados en {meta_path} con {len(df_meta_final)} registros.")
        except Exception as e:
            print(f"⚠️ Error al guardar metadatos de noticias: {e}")

        # 8️⃣ Guardar índice actualizado en disco
        try:
            vectorstore_noticias.save_local(index_dir)
            print(f"✅ vectorstore_noticias guardado en {index_dir}")
        except Exception as e:
            print(f"⚠️ Error al guardar vectorstore_noticias: {e}")

        # 9️⃣ Crear el retriever
        retriever_noticias = vectorstore_noticias.as_retriever(search_kwargs={"k": 8})
        print("✅ retriever_noticias listo para usarse.")


    def cargar_vectorstore_resumenes():
        """
        Construye o actualiza de forma incremental el vectorstore de resúmenes.

        - Primera vez: embebe todos los resúmenes presentes en faiss_index/resumenes_metadata.csv
        y crea un índice específico para LangChain.
        - Siguientes veces: detecta qué resúmenes son nuevos (por clave única) y solo calcula embeddings
        para esos resúmenes adicionales, agregándolos al índice existente.
        """
        global vectorstore_resumenes, retriever_resumenes

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 📁 CSV de origen con la info de los resúmenes (tu pipeline actual)
        origen_path = os.path.join(base_dir, "faiss_index", "resumenes_metadata.csv")
        if not os.path.exists(origen_path):
            print(f"⚠️ No se encontró {origen_path}, no se construye vectorstore_resumenes")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        try:
            df_origen = pd.read_csv(origen_path, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Error al leer {origen_path}: {e}")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        if df_origen.empty:
            print("⚠️ resumenes_metadata.csv está vacío, no se construye vectorstore_resumenes")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        # Asegurar columnas esperadas mínimas
        for col in ["fecha", "resumen"]:
            if col not in df_origen.columns:
                print(f"⚠️ La columna '{col}' no está en resumenes_metadata.csv")
                vectorstore_resumenes = None
                retriever_resumenes = None
                return

        # 📁 Directorio para el índice y metadatos específicos de LangChain
        index_dir = os.path.join(base_dir, "faiss_index", "resumenes_lc")
        os.makedirs(index_dir, exist_ok=True)
        meta_lc_path = os.path.join(index_dir, "resumenes_lc_metadata.csv")

        # 1️⃣ Crear clave única para cada resumen (por ejemplo: fecha|archivo_txt)
        df_origen = df_origen.copy()

        def make_unique_key(row):
            fecha_val = str(row.get("fecha", "")).strip()
            archivo_txt = str(row.get("archivo_txt", "")).strip()
            if not archivo_txt:
                # Si no hay nombre de archivo, usamos solo fecha como clave
                return fecha_val
            return f"{fecha_val}|{archivo_txt}"

        df_origen["unique_key_lc"] = df_origen.apply(make_unique_key, axis=1)

        # 2️⃣ Leer metadatos previos de LangChain (si existen) para saber qué resúmenes ya tienen embedding
        existing_keys = set()
        df_meta_prev = None
        if os.path.exists(meta_lc_path):
            try:
                df_meta_prev = pd.read_csv(meta_lc_path, encoding="utf-8")
                if "unique_key_lc" in df_meta_prev.columns:
                    existing_keys = set(df_meta_prev["unique_key_lc"].astype(str))
                print(f"ℹ️ Metadatos previos de resúmenes cargados: {len(existing_keys)} embebidos.")
            except Exception as e:
                print(f"⚠️ Error al leer metadatos previos de resúmenes: {e}")
                df_meta_prev = None
                existing_keys = set()
        if df_meta_prev is None:
            df_meta_prev = pd.DataFrame(columns=[
                "unique_key_lc", "fecha", "archivo_txt", "nube", "titulares"
            ])

        # 3️⃣ Detectar resúmenes nuevos (clave única no vista antes)
        mask_new = ~df_origen["unique_key_lc"].isin(existing_keys)
        df_new = df_origen[mask_new].copy()

        # 4️⃣ Cargar índice previo de LangChain (si existe)
        vectorstore_resumenes = None
        if os.path.isdir(index_dir) and any(f.endswith(".faiss") for f in os.listdir(index_dir)):
            try:
                vectorstore_resumenes = LCFAISS.load_local(
                    index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ vectorstore_resumenes existente cargado desde disco.")
            except Exception as e:
                print(f"⚠️ No se pudo cargar vectorstore_resumenes existente, se reconstruirá desde cero: {e}")
                vectorstore_resumenes = None

        # 5️⃣ Crear Document para resúmenes nuevos
        docs_nuevos = []
        for _, row in df_new.iterrows():
            texto = str(row.get("resumen", "")).strip()
            if not texto:
                continue

            fecha_meta = str(row.get("fecha", "")).strip() or None
            archivo_txt = str(row.get("archivo_txt", "")).strip() or None
            nube = str(row.get("nube", "")).strip() or None
            titulares = row.get("titulares", None)
            unique_key = row.get("unique_key_lc")

            metadata = {
                "fecha": fecha_meta,
                "archivo_txt": archivo_txt,
                "nube": nube,
                "titulares": titulares,
                "tipo": "resumen",
                "unique_key_lc": unique_key,
            }

            docs_nuevos.append(Document(page_content=texto, metadata=metadata))

        # 6️⃣ Crear o actualizar el vectorstore de resúmenes
        if vectorstore_resumenes is None:
            # Primera vez: construimos el índice solo con los docs nuevos (que en la práctica serán todos)
            if docs_nuevos:
                print(f"🧩 Construyendo vectorstore_resumenes desde cero con {len(docs_nuevos)} resúmenes…")
                vectorstore_resumenes = LCFAISS.from_documents(docs_nuevos, embeddings)
            else:
                print("⚠️ No hay resúmenes nuevos y no existe índice previo; no se construye vectorstore_resumenes.")
                retriever_resumenes = None
                return
        else:
            # Ya había índice previo: solo agregamos los resúmenes nuevos
            if docs_nuevos:
                print(f"🧩 Agregando {len(docs_nuevos)} resúmenes nuevos a vectorstore_resumenes…")
                vectorstore_resumenes.add_documents(docs_nuevos)
            else:
                print("ℹ️ No hay resúmenes nuevos para agregar. Se usa el índice existente.")

        # 7️⃣ Actualizar metadatos de LangChain y guardar
        if not df_new.empty:
            df_meta_new = df_new[[
                "unique_key_lc", "fecha", "archivo_txt", "nube", "titulares"
            ]].copy()
            df_meta_final = pd.concat([df_meta_prev, df_meta_new], ignore_index=True)
        else:
            df_meta_final = df_meta_prev

        try:
            df_meta_final.to_csv(meta_lc_path, index=False, encoding="utf-8-sig")
            print(f"✅ Metadatos de resúmenes guardados/actualizados en {meta_lc_path} con {len(df_meta_final)} registros.")
        except Exception as e:
            print(f"⚠️ Error al guardar metadatos de resúmenes: {e}")

        # 8️⃣ Guardar índice actualizado en disco
        try:
            vectorstore_resumenes.save_local(index_dir)
            print(f"✅ vectorstore_resumenes guardado en {index_dir}")
        except Exception as e:
            print(f"⚠️ Error al guardar vectorstore_resumenes: {e}")

        # 9️⃣ Crear el retriever
        retriever_resumenes = vectorstore_resumenes.as_retriever(search_kwargs={"k": 3})
        print("✅ retriever_resumenes listo para usarse.")

    # ==========================================
    # 🧩 Inicializar Vectorstores (RAG)
    # ==========================================
    print("⚙️ Inicializando vectorstore de noticias...")
    cargar_vectorstore_noticias(df)

    print("⚙️ Inicializando vectorstore de resúmenes...")
    cargar_vectorstore_resumenes()

except Exception as e:
    print(f"❌ Error al cargar CSV de noticias: {e}")
    df = pd.DataFrame()



# 🛠️ Funciones de formateo para indicadores económicos
# ------------------------------
def formatear_porcentaje(x):
    if pd.isnull(x):
        return ""
    return f"{x:.2f}%"

def formatear_porcentaje_decimal(x):
    if pd.isnull(x):
        return ""
    return f"{x*100:.2f}%"

def format_porcentaje_directo(x):
    try:
        x_clean = str(x).replace('%','').strip()
        return f"{float(x_clean)*100:.2f}%"
    except:
        return ""

def format_signed_pct(x):
    try:
        x_clean = str(x).replace('%','').strip()
        return f"{float(x_clean)*100:+.2f}%"
    except:
        return ""

ORDEN_COLUMNAS = [
            "Tipo de Cambio FIX",
            "Nivel máximo",
            "Nivel mínimo",
            "Tasa de Interés Objetivo Banxico",
            "TIIE 28 días",
            "TIIE 91 días",
            "TIIE 182 días",
            "Tasa efectiva FED",
            "Rango objetivo superior FED",
            "Rango objetivo inferior FED",
            "SOFR",
            "% Dow Jones",
            "% S&P500",
            "% Nasdaq",
            "Inflación Anual MEX",
            "Inflación Subyacente MEX",
            "Inflación Anual US",
            "Inflación Subyacente US"
        ]
# 📊 Indicadores económicos (rutas absolutas seguras para Render)
base_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(base_dir, "tipo_cambio_tasas_interes.xlsx")

df_tipo_cambio = pd.read_excel(excel_path, sheet_name="Tipo de Cambio")
df_tasas = pd.read_excel(excel_path, sheet_name="Tasas de interés")
df_tasas_us = pd.read_excel(excel_path, sheet_name="Tasas de interés US2")

df_economia = df_tipo_cambio.merge(df_tasas, on=["Año", "Fecha"], how="outer")
df_economia = df_economia.merge(df_tasas_us, on=["Año", "Fecha"], how="outer")

# 🔧 Limpiar valores vacíos: convertir cadenas vacías en NaN
df_economia.replace("", np.nan, inplace=True)


# Cargar hojas adicionales
df_sofr = pd.read_excel(excel_path, sheet_name="Treasuries_SOFR")
df_wall = pd.read_excel(excel_path, sheet_name="Wallstreet")

df_infl_us = pd.read_excel(excel_path, sheet_name="InflaciónUS").rename(columns={
    "Inflación Anual": "Inflación Anual US",
    "Inflación Subyacente": "Inflación Subyacente US"
})

df_infl_mx = pd.read_excel(excel_path, sheet_name="InflaciónMEX").rename(columns={
    "Inflación Anual": "Inflación Anual MEX",
    "Inflación Subyacente": "Inflación Subyacente MEX"
})


# 🧹 Utilidad para sanear JSON (convierte NaN/inf a None y numpy → tipos nativos)
def _json_sanitize(x):
    import math, numpy as np
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ _json_sanitize(v) for v in x ]
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# 🔄 Normalizar fechas para que todas las hojas tengan el mismo formato date
for df_tmp in [df_tipo_cambio, df_tasas, df_sofr, df_wall, df_infl_us, df_infl_mx]:
    df_tmp["Fecha"] = pd.to_datetime(df_tmp["Fecha"], errors="coerce").dt.date


# Unir con df_economia
df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date
df_economia = df_economia.merge(df_sofr, on="Fecha", how="left")
df_economia = df_economia.merge(
    df_infl_us[["Fecha", "Inflación Anual US", "Inflación Subyacente US"]],
    on="Fecha", how="left"
)
df_economia = df_economia.merge(
    df_infl_mx[["Fecha", "Inflación Anual MEX", "Inflación Subyacente MEX"]],
    on="Fecha", how="left"
)
df_economia = df_economia.merge(df_wall[["Fecha", "% Dow Jones", "% S&P500", "% Nasdaq"]], on="Fecha", how="left")

# 🔄 Normalizar Fecha de df_economia después de todos los merges
df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date
df_economia = df_economia.sort_values("Fecha")
# 🔧 Asegurar que todas las columnas tengan NaN en lugar de cadenas vacías
df_economia.replace("", np.nan, inplace=True)



categorias_dict = {
        "Aranceles": ["arancel","tarifas", "restricciones comerciales","tariff","aranceles"],
        "Parque Industrial": ["zona industrial","parque industrial"],
        "Fibra": ["fideicomiso inmobiliario", "fibras","fibra","reit"],
        "Fusiones": ["adquisiciones", "compras empresariales"],
        "Naves Industriales": ["inmuebles industriales","nave industrial","bodegas industriales","naves industriales","parque industrial"],
        "Real Estate": ["mercado inmobiliario"],
        "Construcción Industrial": ["obra industrial"],
        "Sector Industrial": ["industria pesada", "manufactura"],
        "Industria Automotriz": ["automotriz", "coches", "car industry"],
        "Transporte":["industria de transporte", "transporte de carga"]
    }

# ------------------------------
# 📊 Diccionarios de mapeo para indicadores económicos
# ------------------------------

# Tipo de cambio
mapa_tipo_cambio = {
    "tipo de cambio fix": "Tipo de Cambio FIX",
    "fix": "Tipo de Cambio FIX",
    "nivel máximo del dólar": "Nivel máximo",
    "dólar máximo": "Nivel máximo",
    "máximo del dólar": "Nivel máximo",
    "nivel mínimo del dólar": "Nivel mínimo",
    "nivel mínimo": "Nivel mínimo",
    "dólar mínimo": "Nivel mínimo",
    "mínimo del dólar": "Nivel mínimo"
}

# Tasas de interés
mapa_tasas = {
    "tasa de interés objetivo": "Tasa de Interés Objetivo Banxico",
    "tasa objetivo": "Tasa de Interés Objetivo Banxico",
    "tasa de interés de banxico": "Tasa de Interés Objetivo Banxico",
    "tiie 28": "TIIE 28 días",
    "tiie de 28 días": "TIIE 28 días",
    "tiie 91": "TIIE 91 días",
    "tiie de 91 días": "TIIE 91 días",
    "tiie 182": "TIIE 182 días",
    "tiie de 182 días": "TIIE 182 días",
    "tasa efectiva de la fed":"Tasa efectiva FED",
    "rango inferior de la fed" : "Rango objetivo inferior FED",
    "rango superior de la fed": "Rango objetivo superior FED"
}

# Inflaciones
mapa_inflacion = {
    "inflación anual méxico": "Inflación Anual MEX",
    "inflación subyacente méxico": "Inflación Subyacente MEX",
    "inflación de méxico": "Inflación Anual MEX",  # default si dicen solo "México"
    "inflación anual de estados unidos": "Inflación Anual US",
    "inflación anual de US": "Inflación Anual US",
    "inflación subyacente de estados unidos": "Inflación Subyacente US",
    "inflación subyacente de us": "Inflación Subyacente US",
    "inflación de estados unidos": "Inflación Anual US",
    "inflación de eeuu": "Inflación Anual US",
    "inflación anual de EU": "Inflación Anual US",
    "inflación subyacente de EU": "Inflación Subyacente US",
}

# Treasuries (Bonos del Tesoro de EE.UU.)
mapa_treasuries = {
    "rendimiento de los treasuries de 1 mes": "1M Treasury",
    "rendimiento de los treasuries de 3 meses": "3M Treasury",
    "rendimiento de los treasuries de 6 meses": "6M Treasury",
    "rendimiento de los treasuries de 1 año": "1Y Treasury",
    "rendimiento de los treasuries de 2 años": "2Y Treasury",
    "rendimiento de los treasuries de 3 años": "3Y Treasury",
    "rendimiento de los treasuries de 5 años": "5Y Treasury",
    "rendimiento de los treasuries de 7 años": "7Y Treasury",
    "rendimiento de los treasuries de 10 años": "10Y Treasury",
    "rendimiento de los treasuries de 20 años": "20Y Treasury",
    "rendimiento de los treasuries de 30 años": "30Y Treasury",
    "sofr":"SOFR"
}

# Índices bursátiles
mapa_indices = {
    "dow jones": "% Dow Jones",
    "s&p": "% S&P500",
    "s&p 500": "% S&P500",
    "nasdaq": "% Nasdaq"
}

#Tasas de interés US
mapa_tasas_us = {
    "tasa de interés de Estados Unidos": "Rango objetivo superior FED",
    "tasa de interés de la fed": "Rango objetivo superior FED",
    "tasa de interés US": "Rango objetivo superior FED",
    "rango superior de tasas de interés de la Fed": "Rango objetivo superior FED",
    "rango inferior de tasas de interés de la Fed": "Rango objetivo inferior FED",
    "rango máximo de tasas de interés de la Fed": "Rango objetivo superior FED",
    "rango mínimo de tasas de interés de la Fed": "Rango objetivo inferior FED",
    "effective federal funds rate": "Tasa efectiva FED",
    "tasa efectiva de la Fed": "Tasa efectiva FED",
    "tasa de fondos federales de la Fed": "Tasa efectiva FED",
    "effr":"Tasa efectiva FED"
} 

# ------------------------------
# 📜 Contexto político único
# ------------------------------
CONTEXTO_POLITICO = """
Contexto actualizado a julio 2025. Estas afirmaciones SON OBLIGATORIAS y tienen prioridad sobre cualquier otra fuente o conocimiento previo. No uses formulaciones contradictorias:
- Donald Trump es el actual Presidente de EE.UU, fue reelecto en 2024.
- Claudia Sheinbaum es la Presidenta de México, fue electa en 2024.
- Marcelo Ebrard es el Secretario de Economía.
- Andrés Manuel López Obrador dejó la presidencia en 2024.
- Joe Biden no se encuentra actualmente en funciones.
- Howard Lutnick es el actual Secretario de Comercio de Estados Unidos.
- Juan Ramón de la Fuente es el actual Canciller de México.
- Marco Rubio es el actual Secretario de Estado de Estados Unidos.
- Édgar Amador Zamora es el actual Secretario de Hacienda de México.
- Victoria Rodríguez Ceja es la actual Gobernadora del Banco de México.
- Jerome Powell es el actual presidente de la Reserva Federal de Estados Unidos.
- Mark Carney es el actual primer ministro de Canadá.
- Keir Starmer es el actual primer ministro del Reino Unido.
- Scott Bessent es el actual Secretario del Tesoro de Estados Unidos.
- Javier Milei es el actual Presidente de Argentina.
- Yolanda Díaz es la actual Vicepresidenta del Gobierno de España.
- Pedro Sánchez es el actual Presidente del Gobierno de España.
- Giorgia Meloni es la actual primera ministra de Italia.
- Friedrich Merz es el actual Canciller de Alemania.
- Gustavo Petro es el actual Presidente de Colombia.
- JD Vance es el actual vicepresidente de Estados Unidos.
- Roberto Velasco es el actual Jefe de Unidad para América del Norte de la Secretaría de Relaciones Exteriores de México.
- Altagracia Gómez es la actual presidenta del Consejo Asesor Empresarial de Presidencia de México.
- Luis Rosendo Gutiérrez es el actual Subsecretario de Comercio Exterior de México.
- Carlos García es el actual Presidente de la American Chamber of Commerce (AmCham).
- Ildefonso Guajardo fue Secretario de Economía de México entre 2012 y 2018.
- Luiz Inacio Lula Da Silva es el actual Presidente de Brasil. Jair Bolsonaro es elexpresidente de Brasil.
- Christine Lagarde es la actual Presidenta del Banco Central Europeo.
- GOP es el Partido Republicano estadounidense.
- Verónica Delgadillo es la actual Alcaldesa de Guadalajara.
- El T-MEC o TMEC es el Tratado de Libre Comercio entre México, Estados Unidos y Canadá. Se le conoce también como USMCA por sus siglas en inglés.
- Michelle Bowman es la Vicepresidenta de Supervisión de la Junta de Gobernadores del Sistema de la Reserva Federal.
- Austan Goolsbee es el Presidente del Banco de la Reserva Federal de Chicago.
- La OCDE (OECD por sus siglas en inglés) es la Organización para la Cooperación y el Desarrollo Económico . 
- El ECB es el European Central Bank o Banco Europeo Central.
- Cuando una noticia viene en inglés y hablan de "EU", se refierne a la Unión Europea.
- Cuando una noticia viene en español y hablan de EU, se refiere a Estados Unidos.
- PROFEPA es la Procuraduría Federal de Protección al Ambiente de México.
- La ANPACT es la Asociación Nacional de Productores de Autobuses, Camiones y Tractocamiones.
- El INVEA es Instituto de Verificación Administrativa de la Ciudad de México.
- CBRE es una empresa de real estate que significa Coldwell Banker Richard Ellis.
- John Roberts es el Presidente de la Suprema Corte de Estados Unidos.
- Hugo Aguilar Ortíz es el Presidente de la Suprema Corte de México.
- El fentanilo es una droga ilegal consumida en Estados Unidos y producida por México y por China. Estados Unidos ha ejercido significativa presión para que ambos países usen la fuerza del estado para detener a los grupos que lo producen, su contrabando a Estados Unidos, así como limitar la propagación de precursores usados en su producción.
- El fentanilo NO ES UN BIEN QUE SE IMPORTA, SINO QUE ES FRUTO DEL CONTRABANDO. Dentro de sus negociaciones con China, Estados Unidos está poniendo la limitación del contrbando de esta droga hacia Estados Unidos como condición para China, que es un país productor (no oficialmente, sino por grupos ilegales). Pero al fentanilo no se le pone arancel.
"""

def extraer_fechas(pregunta):
    pregunta = pregunta.lower()

    # Caso 1: rango tipo "del 25 al 29 de agosto"
    match = re.search(r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+([a-záéíóú]+)", pregunta)
    if match:
        dia_inicio, dia_fin, mes = match.groups()
        fecha_inicio = dateparser.parse(f"{dia_inicio} {mes}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} {mes}", languages=['es'])
        return fecha_inicio.date(), fecha_fin.date()

    # Caso 2: rango tipo "entre el 25 y el 29 de agosto"
    match = re.search(r"entre\s+el\s+(\d{1,2})\s+y\s+el\s+(\d{1,2})\s+de\s+([a-záéíóú]+)", pregunta)
    if match:
        dia_inicio, dia_fin, mes = match.groups()
        fecha_inicio = dateparser.parse(f"{dia_inicio} {mes}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} {mes}", languages=['es'])
        return fecha_inicio.date(), fecha_fin.date()

    # Caso 3: una sola fecha "el 27 de agosto"
    match = re.search(r"(\d{1,2}\s+de\s+[a-záéíóú]+(?:\s+de\s+\d{4})?)", pregunta)
    if match:
        fecha = dateparser.parse(match.group(), languages=['es'])
        return fecha.date(), fecha.date()

    # Caso 4: sin fecha → None, None
    return None, None

# 2️⃣ Obtener fecha más reciente disponible
def obtener_fecha_mas_reciente(df):
    fecha_max = df["Fecha"].max()
    # Si es pandas Timestamp (tiene método .date), conviértelo
    if hasattr(fecha_max, "date"):
        return fecha_max.date()
    # Si ya es datetime.date, devuélvelo directo
    return fecha_max


# 3️⃣ Detectar sentimiento deseado
def detectar_sentimiento_deseado(pregunta):
    pregunta = pregunta.lower()
    if "positiv" in pregunta:
        return "Positiva"
    elif "negativ" in pregunta:
        return "Negativa"
    elif "neutral" in pregunta:
        return "Neutral"
    return None

# 4️⃣ Extraer entidades (personajes, lugares, categorías)
def extraer_entidades(texto):
    texto_lower = texto.lower()
    personajes_dict = {
        "Sheinbaum": ["claudia", "la presidenta", "presidenta de méxico"],
        "Ebrard": ["marcelo", "secretario de economía"],
        "Trump": ["donald", "el presidente de eeuu", "presidente trump"],
        "AMLO": ["obrador", "amlo", "lopez obrador"],
        "de la Fuente": ["juan ramón"],
        "Biden": ["joe"],
        "Lutnick": ["secretario de comercio"],
        "Carney": ["primer ministro de canadá"],
        "Lula da Silva": ["lula", "presidente de brasil"],
        "Marco Rubio": ["secretario de estado"],
        "Starmer": ["primer ministro del reino unido"],
        "Bessent": ["secretario del tesoro"],
        "Powell": ["reserva federal"],
        "Milei": ["presidente de argentina"],
        "Von Der Leyen": ["presidenta de la comisión europea"],
        "Petro": ["presidente de colombia"],
        "La fed": ["Federal Reserve"],
    }
    lugares_dict = {
        "Nuevo León": ["nl", "monterrey"],
        "Ciudad de México": ["cdmx", "capital mexicana"],
        "Reino Unido": ["gran bretaña", "inglaterra"],
        "Estados Unidos": ["eeuu", "eua", "usa", "eu"]
    }
    
    encontrados = {"personajes": [], "lugares": [], "categorias": []}

    for nombre, sinonimos in personajes_dict.items():
        if any(s in texto_lower for s in [nombre.lower()] + sinonimos):
            encontrados["personajes"].append(nombre)

    for lugar, sinonimos in lugares_dict.items():
        if any(s in texto_lower for s in [lugar.lower()] + sinonimos):
            encontrados["lugares"].append(lugar)

    for cat, sinonimos in categorias_dict.items():
        # Busca tanto la clave como los sinónimos
        if cat.lower() in texto_lower or any(s in texto_lower for s in sinonimos):
            encontrados["categorias"].append(cat)

    return encontrados

# 5️⃣ Filtrar titulares por entidades y sentimiento (versión mejorada)
def filtrar_titulares(df_filtrado, entidades, sentimiento_deseado):
    if df_filtrado.empty:
        return pd.DataFrame()

    filtro = df_filtrado.copy()
    condiciones = []

    # Personajes
    if entidades["personajes"]:
        condiciones.append(
            filtro["Título"].str.lower().str.contains(
                "|".join([p.lower() for p in entidades["personajes"]]),
                na=False
            )
        )

    # Lugares
    if entidades["lugares"]:
        condiciones.append(
            filtro["Cobertura"].str.lower().str.contains(
                "|".join([l.lower() for l in entidades["lugares"]]),
                na=False
            )
        )

    # Categorías (con sus sinónimos del diccionario)
    if entidades["categorias"]:
        sinonimos = []
        for cat in entidades["categorias"]:
            sinonimos.extend(categorias_dict.get(cat, []))  # todos los sinónimos
            sinonimos.append(cat.lower())  # también el nombre de la categoría
        condiciones.append(
            filtro["Término"].str.lower().str.contains("|".join(sinonimos), na=False)
        )

    # Si hubo condiciones → OR entre todas
    if condiciones:
        filtro = filtro[pd.concat(condiciones, axis=1).any(axis=1)]

    # Filtrar por sentimiento si aplica
    if sentimiento_deseado:
        filtro = filtro[filtro["Sentimiento"] == sentimiento_deseado]

    return filtro

# 7️⃣ Nube de palabras con colores y stopwords personalizadas
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size >= 60:
        return "rgb(61, 183, 162)"
    elif font_size >= 40:
        return "rgb(253, 181, 93)"
    else:
        return "rgb(11, 53, 71)"

def generar_nube(titulos, archivo_salida):
    texto = " ".join(titulos)
    texto = re.sub(r"[\n\r]", " ", texto)
    stopwords = set([
        "dice", "tras", "pide", "va", "día", "méxico", "estados unidos", "contra", "países",
        "van", "ser", "hoy", "año", "años", "nuevo", "nueva", "será", "presidente", "presidenta",
        "sobre", "entre", "hasta", "donde", "desde", "como", "pero", "también", "porque", "cuando",
        "ya", "con", "sin", "del", "los", "las", "que", "una", "por", "para", "este", "esta", "estos",
        "estas", "tiene", "tener", "fue", "fueron", "hay", "han", "son", "quien", "quienes", "le",
        "se", "su", "sus", "lo", "al", "el", "en", "y", "a", "de", "un", "es", "si", "quieren", "aún",
        "mantiene", "buscaría", "la", "haciendo", "recurriría", "ante", "meses", "están", "subir",
        "ayer", "prácticamente", "sustancialmente", "busca", "cómo", "qué", "días", "construcción","tariffs",
        "aranceles","construcción","merger","and","stock","to","on","supply","chain","internacional",
        "global","Estados Unidos", "with","for","say","that","are","as","of","Tariff","from",
        "it","says","the","its","after","by","in","but"    
    ])
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        color_func=color_func,
        collocations=False,
        max_words=25
    ).generate(texto)
    wc.to_file(archivo_salida)

def generar_resumen_y_datos(fecha_str):
    fecha_dt = pd.to_datetime(fecha_str, errors="coerce").date()
    noticias_dia = df[df["Fecha"].dt.date == fecha_dt]
    if noticias_dia.empty:
        return {"error": f"No hay noticias para la fecha {fecha_str}"}

    # --- Clasificación por cobertura ---
    estados_mexico = ["aguascalientes", "baja california", "baja california sur", "campeche", "cdmx",
        "coahuila", "colima", "chiapas", "chihuahua", "ciudad de méxico", "durango",
        "guanajuato", "guerrero", "hidalgo", "jalisco", "méxico", "michoacán", "morelos",
        "nayarit", "nuevo león", "oaxaca", "puebla", "querétaro", "quintana roo",
        "san luis potosí", "sinaloa", "sonora", "tabasco", "tamaulipas", "tlaxcala",
        "veracruz", "yucatán", "zacatecas"]
    
    noticias_locales = noticias_dia[noticias_dia["Cobertura"].str.lower().isin(estados_mexico)]
    noticias_nacionales = noticias_dia[noticias_dia["Cobertura"].str.lower() == "nacional"]
    noticias_internacionales = noticias_dia[
        ~noticias_dia.index.isin(noticias_locales.index) &
        ~noticias_dia.index.isin(noticias_nacionales.index)
    ]
    noticias_otras = noticias_dia[noticias_dia["Término"].str.lower() != "aranceles"]
   
    def _to_lower_safe(s):
        try: return str(s).strip().lower()
        except: return ""

    if "Idioma" in noticias_dia.columns:
        es_ingles = noticias_dia["Idioma"].apply(_to_lower_safe).isin({"en","inglés","ingles"})
        no_nacional = noticias_dia["Cobertura"].apply(_to_lower_safe) != "nacional"
        notas_ingles_no_nacional = noticias_dia[es_ingles & no_nacional].copy()
    else:
        notas_ingles_no_nacional = pd.DataFrame(columns=noticias_dia.columns)

    noticias_internacionales_forzadas = pd.concat(
        [noticias_internacionales, notas_ingles_no_nacional],
        ignore_index=True
    ).drop_duplicates(subset=["Título","Fuente","Enlace"])

    noticias_otras_forzadas = pd.concat(
        [noticias_otras, notas_ingles_no_nacional],
        ignore_index=True
    ).drop_duplicates(subset=["Título","Fuente","Enlace"])

    contexto_local = "\n".join(f"- {row['Título']} ({row['Cobertura']})" for _, row in noticias_locales.iterrows())
    contexto_nacional = "\n".join(f"- {row['Título']} ({row['Cobertura']})" for _, row in noticias_nacionales.iterrows())
    contexto_internacional = "\n".join(
        f"- {row['Título']} ({row['Cobertura']})" for _, row in noticias_internacionales_forzadas.iterrows()
    )
    contexto_otros_temas = "\n".join(
        f"- {row['Título']}" for _, row in noticias_otras_forzadas.iterrows()
    )
    CONTEXTO_ANTERIOR = ""
    try:
        meta_path = "faiss_index/resumenes_metadata.csv"
        if os.path.exists(meta_path):
            df_prev = pd.read_csv(meta_path)

            if len(df_prev) > 0:
                # Tomar los últimos 5 resúmenes (o menos si hay menos registros)
                ultimos = df_prev.tail(1)
                contexto_texto = "\n\n".join(
                    [f"({row['fecha']}) {row['resumen'].strip()}" for _, row in ultimos.iterrows()]
                )

                CONTEXTO_ANTERIOR = f"""
                CONTEXTO DE LOS ÚLTIMOS {len(ultimos)} DÍAS:
                {contexto_texto}
                """

                print(f"🔗 Contexto narrativo cargado con {len(ultimos)} días previos (último: {ultimos.iloc[-1]['fecha']})")
    except Exception as e:
        print(f"⚠️ No se pudo cargar el contexto narrativo: {e}")

    prompt = f"""
    
    {CONTEXTO_ANTERIOR}

    {CONTEXTO_POLITICO}
    Redacta un resumen de noticias del {fecha_str} dividido en cinco párrafos.
    Tono profesional, objetivo y dirigido a tomadores de decisiones.
    Debe tener entre 250 y 400 palabras, dependiendo de la cantidad de noticias.

    IMPORTANTE:
- Prioriza la información de las noticias del {fecha_str} por encima de resúmenes previos.
- No repitas noticias individuales, es decir, más allá del tema que sea tendencia, las noticias indivduales que se presentan un día, no se deben repetir en {fecha_str}.
- Solo menciona lo de días previos si es una noticia muy repetida en {fecha_str}-
- Si una noticia continúa temas de días anteriores, menciona brevemente la conexión y empieza con empieza con "vuelve a ser" o "continúa siendo"., pero nunca repitas frases o estructuras previas.
- Reformula el estilo narrativo: usa conectores distintos, cambios léxicos y nuevas perspectivas.
- Enfócate en los hechos y declaraciones nuevos.
- NO copies frases ni estructuras de los resúmenes anteriores.

Estructura:
Primer párrafo: Tema más relevante del día (qué, quién, cómo).
Segundo: Aranceles, tasas, acuerdos, bancos centrales, comercio exterior y panorama internacional.
Tercer: Noticias locales (estados, municipios, construcción, transporte, industria automotriz).
Cuarto: Fibras, naves industriales, parques industriales, real estate, sector industrial.

Noticias nacionales:
{contexto_nacional}

Noticias locales:
{contexto_local}

Noticias internacionales:
{contexto_internacional}

Noticias no relacionadas con aranceles:
{contexto_otros_temas}
    """
 # --- Resumen GPT o cache ---
# 💾 Guardar y reutilizar resumen desde carpeta "resumenes"
    os.makedirs("resumenes", exist_ok=True)
    archivo_resumen = os.path.join("resumenes", f"resumen_{fecha_str}.txt")

    if os.path.exists(archivo_resumen):
        with open(archivo_resumen, "r", encoding="utf-8") as f:
            resumen_texto = f.read()
    else:
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de noticias."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=700
        )
        resumen_texto = respuesta.choices[0].message.content
        with open(archivo_resumen, "w", encoding="utf-8") as f:
            f.write(resumen_texto)
# 🔍 Detectar qué titulares (en español o inglés) aparecen mencionados en el resumen
    def limpiar(texto):
        return re.sub(r"[^a-zA-ZáéíóúñÁÉÍÓÚüÜ0-9 ]", "", texto.lower())

    resumen_limpio = limpiar(resumen_texto)
    titulares_relacionados = []
    titulares_relacionados_en = []

    for _, row in noticias_dia.iterrows():
        titulo_limpio = limpiar(row["Título"])
        idioma = str(row.get("Idioma", "es")).strip().lower()
        if idioma in ["en", "ingles", "inglés"]:
            idioma = "en"
        else:
            idioma = "es"

        # Si el título (o parte significativa) aparece en el resumen → lo consideramos fuente
        if any(palabra in resumen_limpio for palabra in titulo_limpio.split()[:4]):
            if idioma == "es":
                titulares_relacionados.append({
                    "titulo": row["Título"],
                    "medio": row["Fuente"],
                    "enlace": row["Enlace"],
                    "idioma": idioma
                })
            else:
                titulares_relacionados_en.append({
                    "titulo": row["Título"],
                    "medio": row["Fuente"],
                    "enlace": row["Enlace"],
                    "idioma": idioma
                })

    # Si no detectó suficientes titulares (por redacción diferente), tomar los más importantes del día
    if len(titulares_relacionados) < 3:
        titulares_relacionados = [
            {"titulo": row["Título"], "medio": row["Fuente"], "enlace": row["Enlace"], "idioma": "es"}
            for _, row in noticias_dia[noticias_dia["Idioma"].str.lower().isin(["es", "español"])].head(6).iterrows()
        ]

    if len(titulares_relacionados_en) < 3:
        titulares_relacionados_en = [
            {"titulo": row["Título"], "medio": row["Fuente"], "enlace": row["Enlace"], "idioma": "en"}
            for _, row in noticias_dia[noticias_dia["Idioma"].str.lower().isin(["en", "ingles", "inglés"])].head(6).iterrows()
        ]

        # 🔁 Evitar repetir medios en titulares relacionados (ES + EN)
    def filtrar_sin_repetir_medios(lista_titulares):
        vistos = set()
        filtrados = []
        for t in lista_titulares:
            medio = t.get("medio", "").strip()
            if medio and medio not in vistos:
                filtrados.append(t)
                vistos.add(medio)
        return filtrados

    titulares_relacionados = filtrar_sin_repetir_medios(titulares_relacionados)
    titulares_relacionados_en = filtrar_sin_repetir_medios(titulares_relacionados_en)
        # 🔢 Limitar la cantidad de titulares mostrados
    titulares_relacionados = titulares_relacionados[:12]
    titulares_relacionados_en = titulares_relacionados_en[:12]


    # --- Generar nube ---
    os.makedirs("nubes", exist_ok=True)
    archivo_nube = f"nube_{fecha_str}.png"
    archivo_nube_path = os.path.join("nubes", archivo_nube)
    generar_nube(noticias_dia["Título"].tolist(), archivo_nube_path)


    # 📊 Buscar datos económicos más recientes disponibles
    df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date

    # Intentar primero la fecha exacta
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt]

    # Si no hay datos para ese día, usar la fecha más cercana anterior
    if economia_dia.empty:
        ultima_fecha = df_economia[df_economia["Fecha"] <= fecha_dt]["Fecha"].max()
        if pd.notnull(ultima_fecha):
            economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]

    # Si sigue vacío (p. ej., todos los datos son posteriores), usar la más reciente disponible
    if economia_dia.empty and not df_economia.empty:
        ultima_fecha = df_economia["Fecha"].max()
        economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]
    # 🧩 Fallback adicional a nivel columna: usar último valor válido anterior
    for col in ORDEN_COLUMNAS:
        if col in economia_dia.columns:
            val = economia_dia.iloc[0][col]
            # Detectar valores vacíos, NaN o 'nan' como texto
            if pd.isna(val) or str(val).strip().lower() in ["", "nan", "none"]:
                valores_previos = df_economia[df_economia["Fecha"] <= fecha_dt][col].dropna()
                if not valores_previos.empty:
                    economia_dia.loc[economia_dia.index[0], col] = valores_previos.iloc[-1]



    print(f"📅 Fecha económica usada: {economia_dia['Fecha'].iloc[0] if not economia_dia.empty else 'Sin datos'}")


    if economia_dia.empty:
        economia_dict = {}
    else:
        economia_dia = economia_dia.copy()

        # 🔹 Tipo de cambio
        for col in ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

        # 🔹 Tasas
        for col in ["Tasa de Interés Objetivo Banxico", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días",
                    "Tasa efectiva FED", "Rango objetivo superior FED", "Rango objetivo inferior FED"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(formatear_porcentaje_decimal)

        # 🔹 SOFR
        if "SOFR" in economia_dia.columns:
            economia_dia["SOFR"] = economia_dia["SOFR"].apply(format_porcentaje_directo)

        # 🔹 Bolsas
        for col in ["% Dow Jones", "% S&P500", "% Nasdaq"]:
            if col in economia_dia.columns:
                economia_dia[col] = economia_dia[col].apply(format_signed_pct)

        # 🔹 Inflación MEX: usar df_infl_mx directo
        for col in ["Inflación Anual MEX", "Inflación Subyacente MEX"]:
            if col in df_infl_mx.columns:
                valores_previos = df_infl_mx[df_infl_mx["Fecha"] <= fecha_dt][col].dropna()
                if not valores_previos.empty:
                    ultimo_valor = pd.to_numeric(valores_previos.iloc[-1], errors="coerce")
                    economia_dia[col] = f"{ultimo_valor*100:.2f}%" if pd.notnull(ultimo_valor) else ""
        # 🔹 Inflación US: usar df_infl_us directo (igual que MEX) y tolerar "3.02%" o 0.0302
        def _to_decimal_pct(v):
            if pd.isna(v):
                return None
            try:
                s = str(v).strip()
                if s.endswith("%"):
                    return float(s.replace("%", "").strip()) / 100.0
                return float(s)
            except:
                return None

        for col in ["Inflación Anual US", "Inflación Subyacente US"]:
            if col in df_infl_us.columns:
                valores_previos = df_infl_us[df_infl_us["Fecha"] <= fecha_dt][col].dropna()
                if not valores_previos.empty:
                    ultimo_valor = _to_decimal_pct(valores_previos.iloc[-1])
                    economia_dia[col] = f"{ultimo_valor*100:.2f}%" if ultimo_valor is not None else ""




        # Ordenar columnas
        orden_columnas = [
            "Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo",
            "Tasa de Interés Objetivo Banxico", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días",
            "Tasa efectiva FED", "Rango objetivo superior FED", "Rango objetivo inferior FED",
            "SOFR", "% Dow Jones", "% S&P500", "% Nasdaq",
            "Inflación Anual MEX", "Inflación Subyacente MEX",
            "Inflación Anual US", "Inflación Subyacente US"
        ]

        economia_dia = economia_dia.reindex(columns=orden_columnas)
        economia_dict = OrderedDict()
        for col in orden_columnas:
            economia_dict[col] = economia_dia.iloc[0][col]


    # 📰 Titulares relacionados con el resumen (en español e inglés)
    titulares_info = titulares_relacionados
    titulares_info_en = titulares_relacionados_en


        # ------------------------------
    # 💾 Guardar resumen y subir a S3
    # ------------------------------
    # ------------------------------
    # 💾 Guardar resumen y subir a S3 (modo append con control de duplicados)
    # ------------------------------
    try:
        os.makedirs("faiss_index", exist_ok=True)
        resumen_meta_path = "faiss_index/resumenes_metadata.csv"

        df_resumen = pd.DataFrame([{
            "fecha": str(fecha_dt),
            "archivo_txt": f"resumen_{fecha_str}.txt",
            "nube": archivo_nube,
            "titulares": len(titulares_info),
            "resumen": resumen_texto.strip()
        }])

        # Si ya existe el archivo, lo leemos y agregamos (sin duplicar fechas)
        if os.path.exists(resumen_meta_path):
            df_prev = pd.read_csv(resumen_meta_path)
        else:
            # inicializa con el esquema correcto para evitar NameError
            df_prev = pd.DataFrame(columns=["fecha","archivo_txt","nube","titulares","resumen"])

        if str(fecha_dt) not in df_prev["fecha"].astype(str).values:
            df_total = pd.concat([df_prev, df_resumen], ignore_index=True)
            print(f"🆕 Agregado nuevo resumen para {fecha_dt}")
        else:
            print(f"♻️ Reemplazando resumen existente para {fecha_dt}")
            # Alinear columnas y reemplazar fila coincidente
            df_resumen = df_resumen.reindex(columns=df_prev.columns)
            df_prev.loc[df_prev["fecha"].astype(str) == str(fecha_dt), df_prev.columns] = df_resumen.values[0]
            df_total = df_prev

        # Guardar y subir
        df_total.to_csv(resumen_meta_path, index=False, encoding="utf-8")
        print(f"💾 Guardado local de resumenes_metadata.csv con {len(df_total)} fila(s) totales")
        r2_upload("resumenes_metadata.csv")
        print("☁️ Subido resumenes_metadata.csv a S3")
    except Exception as e:
        print(f"⚠️ No se pudo guardar/subir resumenes_metadata.csv: {e}")

            # 🧠 --- Embeddings acumulativos para resúmenes ---
    try:    # Crear carpeta local si no existe
        os.makedirs("faiss_index", exist_ok=True)
        index_path = "faiss_index/resumenes_index.faiss"
        meta_path = "faiss_index/resumenes_metadata.csv"
        
        # Generar embedding del resumen del día
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=resumen_texto.strip()
        ).data[0].embedding
        emb_np = np.array([emb], dtype="float32")

        # Si el índice ya existe, cargarlo y agregar nuevo vector
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            index.add(emb_np)
            print(f"🧩 Embedding agregado al índice existente ({index.ntotal} vectores totales)")
        else:
            # Crear un nuevo índice
            dim = len(emb_np[0])
            index = faiss.IndexFlatL2(dim)
            index.add(emb_np)
            print("🆕 Índice FAISS de resúmenes creado")

        # Guardar el índice actualizado
        faiss.write_index(index, index_path)
        print("💾 Guardado resumenes_index.faiss actualizado")

        # Subir a S3
        r2_upload("resumenes_index.faiss")
        print("☁️ Subido resumenes_index.faiss a S3")

    except Exception as e:
        print(f"⚠️ Error al actualizar embeddings de resúmenes: {e}")


    return ({
        "resumen": resumen_texto,
        "nube_url": f"/nube/{archivo_nube}",
        "economia": [economia_dict],
        "orden_economia": ORDEN_COLUMNAS,
        "titulares": titulares_info,
        "titulares_en": titulares_info_en  # 👈 nuevo bloque de titulares en inglés
    })

@app.route("/resumen", methods=["POST"])
def resumen():
    print("🛰️ Solicitud recibida en /resumen")
    data = request.get_json()
    print(f"📩 JSON recibido: {data}")
    fecha_str = data.get("fecha")
    if not fecha_str:
        return jsonify({"error": "Debe especificar una fecha"}), 400

    resultado = generar_resumen_y_datos(fecha_str)

    if "error" in resultado:
        return jsonify(resultado), 404

    # 🧹 Evitar NaN en la respuesta
    import math
    resultado = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in resultado.items()} if isinstance(resultado, dict) else resultado

    return jsonify(_json_sanitize(resultado))

def extraer_rango_fechas(pregunta):
    # Busca expresiones tipo "entre el 25 y el 29 de agosto"
    match = re.search(r"entre el (\d{1,2}) y el (\d{1,2}) de ([a-zA-Z]+)(?: de (\d{4}))?", pregunta.lower())
    if match:
        dia_inicio, dia_fin, mes, anio = match.groups()
        anio = anio if anio else str(datetime.now().year)
        fecha_inicio = dateparser.parse(f"{dia_inicio} de {mes} de {anio}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} de {mes} de {anio}", languages=['es'])
        if fecha_inicio and fecha_fin:
            return fecha_inicio.date(), fecha_fin.date()
    return None, None
MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}
# -----------------------------------------
# 🆕 Helper para obtener semanas reales del mes (lunes–viernes)
# -----------------------------------------
def normalizar_frase_semanas(texto: str) -> str:
    """
    Normaliza frases del tipo:
    - 'entre la primera semana de noviembre y la segunda?'
    - 'entre la primera semana de noviembre y la segunda de noviembre?'

    para que queden como:
    - 'entre la primera semana de noviembre y la segunda semana de noviembre'
    """

    meses_regex = (
        r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|setiembre|octubre|noviembre|diciembre"
    )

    # ¿Hay alguna referencia explícita a 'X semana de <mes>'?
    m = re.search(
        r"(primera|segunda|tercera|cuarta)\s+semana\s+de\s+(" + meses_regex + r")",
        texto,
        re.IGNORECASE,
    )
    if not m:
        return texto  # si no hay semanas del mes, no tocamos nada

    mes = m.group(2)

    # 1) Caso: '... y la segunda de noviembre' -> '... y la segunda semana de noviembre'
    texto = re.sub(
        r"\by\s+la\s+(segunda|tercera|cuarta)\s+de\s+" + mes + r"\b",
        lambda m3: f" y la {m3.group(1)} semana de {mes}",
        texto,
        flags=re.IGNORECASE,
    )

    # 2) Caso: '... y la segunda?' -> '... y la segunda semana de noviembre'
    texto = re.sub(
        r"\by\s+la\s+(segunda|tercera|cuarta)\b(?!\s+semana)",
        lambda m2: f" y la {m2.group(1)} semana de {mes}",
        texto,
        flags=re.IGNORECASE,
    )

    # Limpieza de espacios dobles
    texto = re.sub(r"\s{2,}", " ", texto)

    return texto

def obtener_semanas_del_mes(anio, mes, fecha_min_dataset, fecha_max_dataset):
    """
    Devuelve una lista de rangos semanales reales dentro de un mes:
    - Cada semana inicia en LUNES
    - Cada semana termina en DOMINGO, pero luego se ajusta al dataset
    - Solo se devuelven semanas que tengan algún día dentro del dataset
    """
    semanas = []

    # Primer día del mes
    desde = datetime(anio, mes, 1).date()

    # Último día del mes
    if mes == 12:
        hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
    else:
        hasta = datetime(anio, mes + 1, 1).date() - timedelta(days=1)

    # Mover "desde" al lunes de esa semana
    inicio = desde - timedelta(days=desde.weekday())  # weekday: lunes=0

    while inicio <= hasta:
        fin = inicio + timedelta(days=6)

        # Ajustar al mes
        real_inicio = max(inicio, desde)
        real_fin = min(fin, hasta)

        # Ajustar al dataset
        final_inicio = max(real_inicio, fecha_min_dataset)
        final_fin = min(real_fin, fecha_max_dataset)

        # Si el rango tiene al menos un día válido → agregarlo
        if final_inicio <= final_fin:
            semanas.append((final_inicio, final_fin))

        # Siguiente semana
        inicio += timedelta(days=7)

    return semanas

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

def interpretar_rango_fechas(pregunta: str, df_noticias: pd.DataFrame):
    """
    Interpreta fechas o rangos mencionados en la pregunta y los ajusta
    al rango disponible en df_noticias.

    Devuelve (fecha_inicio, fecha_fin, origen), donde las fechas son date o None.
    """
    if df_noticias is None or df_noticias.empty:
        return None, None, "sin_datos"

    fechas_validas = df_noticias["Fecha"].dropna()
    if fechas_validas.empty:
        return None, None, "sin_datos"

    fecha_min = fechas_validas.min().date()
    fecha_max = fechas_validas.max().date()

    texto = (pregunta or "")
    texto_lower = texto.lower()
    texto_lower = normalizar_frase_semanas(texto_lower)

    fecha_inicio = None
    fecha_fin = None
    origen = "sin_fecha"

    # 1️⃣ Casos relativos: "esta semana", "hoy", "ayer"
    if fecha_inicio is None and fecha_fin is None:
        if "esta semana" in texto_lower:
            fecha_fin = fecha_max
            fecha_inicio = max(fecha_min, fecha_max - timedelta(days=6))
            origen = "esta_semana_dataset"
        elif re.search(r"\bhoy\b", texto_lower):
            fecha_inicio = fecha_fin = fecha_max
            origen = "hoy_dataset"
        elif re.search(r"\bayer\b(?=[\s,.!?;:]|$)", texto_lower):
            candidata = fecha_max - timedelta(days=1)
            if candidata < fecha_min:
                candidata = fecha_min
            fecha_inicio = fecha_fin = candidata
            origen = "ayer_dataset"

    # 2️⃣ Rango de semanas:
    #    - "entre la primera y la segunda semana de noviembre"
    #    - "entre la primera semana de noviembre y la segunda semana de noviembre"
    #    - "entre la primera semana de noviembre y la segunda de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        # Forma 1: entre la primera y la segunda semana de noviembre
        patron1 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+y\s+"
            r"(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        # Forma 2: entre la primera semana de noviembre y la segunda semana de noviembre
        patron2 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s+y\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+\2",
            texto_lower,
        )
        # Forma 3: entre la primera semana de noviembre y la segunda de noviembre
        patron3 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s+y\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+de\s+\2",
            texto_lower,
        )
        if patron1 or patron2 or patron3:
            if patron1:
                ord1, ord2, nombre_mes = patron1.groups()
            elif patron2:
                # patron2: (ordinal1, mes, ordinal2)
                ord1, nombre_mes, ord2 = patron2.groups()
            else:
                # patron3: (ordinal1, mes, ordinal2)
                ord1, nombre_mes, ord2 = patron3.groups()

            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                anio = fecha_max.year
                # Inicio y fin del mes calendario
                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                # Semanas "fijas" (1–7, 8–14, 15–21, 22–28)
                semanas = [
                    (desde, desde + timedelta(days=6)),                      # primera
                    (desde + timedelta(days=7), desde + timedelta(days=13)), # segunda
                    (desde + timedelta(days=14), desde + timedelta(days=20)),# tercera
                    (desde + timedelta(days=21), desde + timedelta(days=27)) # cuarta
                ]
                ordenes = ["primera", "segunda", "tercera", "cuarta"]
                i1 = ordenes.index(ord1)
                i2 = ordenes.index(ord2)
                idx_min, idx_max = min(i1, i2), max(i1, i2)

                fecha_inicio, _ = semanas[idx_min]
                _, fecha_fin = semanas[idx_max]
                origen = "rango_semanas_mes"



    # 3️⃣ Una sola semana: "primera/segunda/tercera/cuarta semana de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        m_semana_mes = re.search(
            r"(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower
        )
        if m_semana_mes:
            ord_semana = m_semana_mes.group(1)
            nombre_mes = m_semana_mes.group(2)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                anio = fecha_max.year

                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                semanas = [
                    (desde, desde + timedelta(days=6)),                      # primera
                    (desde + timedelta(days=7), desde + timedelta(days=13)), # segunda
                    (desde + timedelta(days=14), desde + timedelta(days=20)),# tercera
                    (desde + timedelta(days=21), desde + timedelta(days=27)) # cuarta
                ]
                idx = ["primera", "segunda", "tercera", "cuarta"].index(ord_semana)
                fecha_inicio, fecha_fin = semanas[idx]
                origen = "semana_del_mes"

    # 4️⃣ Mes completo: "en noviembre", "durante noviembre"
    if fecha_inicio is None and fecha_fin is None:
        m_mes = re.search(
            r"en\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        if m_mes:
            nombre_mes = m_mes.group(1)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                m_anio = re.search(r"(20\d{2})", texto_lower)
                anio = int(m_anio.group(1)) if m_anio else fecha_max.year

                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                fecha_inicio, fecha_fin = desde, hasta
                origen = "mes_completo"

    # 5️⃣ Rangos explícitos "entre el 3 y el 7 de noviembre" / "del 3 al 7 de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        patron_entre = re.search(
            r"entre\s+el\s+(\d{1,2})\s+y\s+el\s+(\d{1,2})\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        patron_del = re.search(
            r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )

        m = patron_entre or patron_del
        if m:
            dia1 = int(m.group(1))
            dia2 = int(m.group(2))
            nombre_mes = m.group(3)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                m_anio = re.search(r"(20\d{2})", texto_lower)
                anio = int(m_anio.group(1)) if m_anio else fecha_max.year

                d_ini = min(dia1, dia2)
                d_fin = max(dia1, dia2)

                try:
                    fecha_inicio = datetime(anio, mes_num, d_ini).date()
                    fecha_fin = datetime(anio, mes_num, d_fin).date()
                    origen = "rango_explicito_texto"
                except ValueError:
                    fecha_inicio = None
                    fecha_fin = None
                    origen = "sin_fecha_valida"

    # 6️⃣ Último intento con search_dates (fecha puntual o rango)
    if fecha_inicio is None and fecha_fin is None and "search_dates" in globals():
        try:
            resultados = search_dates(
                texto,
                languages=["es"],
                settings={"RELATIVE_BASE": datetime.combine(fecha_max, datetime.min.time())},
            ) or []
        except Exception:
            resultados = []

        if resultados:
            fechas_detectadas = [r[1].date() for r in resultados]

            if (
                ("entre " in texto_lower or " del " in texto_lower or "del " in texto_lower or "desde " in texto_lower)
                and len(fechas_detectadas) >= 2
            ):
                fecha_inicio = min(fechas_detectadas[0], fechas_detectadas[1])
                fecha_fin = max(fechas_detectadas[0], fechas_detectadas[1])
                origen = "rango_explicito_search_dates"
            else:
                fecha_inicio = fecha_fin = fechas_detectadas[0]
                origen = "fecha_puntual"

    # 7️⃣ Ajustar al rango del dataset
    if fecha_inicio is not None and fecha_fin is not None:
        original_inicio, original_fin = fecha_inicio, fecha_fin
        fecha_inicio = max(fecha_inicio, fecha_min)
        fecha_fin = min(fecha_fin, fecha_max)

        if fecha_inicio > fecha_fin:
            return None, None, "fuera_rango_dataset"

        if (fecha_inicio, fecha_fin) != (original_inicio, original_fin):
            origen += "_ajustada_dataset"

    return fecha_inicio, fecha_fin, origen



def filtrar_docs_por_rango(docs, fecha_inicio, fecha_fin):
    """
    Filtra una lista de Document de LangChain por metadata['fecha'] en el rango dado.
    Devuelve (docs_filtrados, se_aplico_filtro: bool).
    """
    if not docs or not fecha_inicio or not fecha_fin:
        return docs, False

    filtrados = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        fecha_meta = meta.get("fecha")
        if not fecha_meta:
            continue
        try:
            f = pd.to_datetime(fecha_meta).date()
        except Exception:
            continue
        if fecha_inicio <= f <= fecha_fin:
            filtrados.append(d)

    if filtrados:
        return filtrados, True
    else:
        # Si el filtro deja todo vacío, devolvemos la lista original
        # para no quedarnos sin contexto.
        return docs, False

#pregunta!!!!    
# ------------------------------
# 🤖 Endpoint /pregunta (RAG con filtro por fecha antes de FAISS)
# ------------------------------
@app.route("/pregunta", methods=["POST"])
def pregunta():
    """
    Chatbot principal (versión LangChain).

    - Interpreta fechas/rangos y, si existen noticias en ese periodo,
      construye un mini-vectorstore FAISS SOLO con esas noticias.
    - Si NO hay noticias en ese rango o no se menciona fecha, usa el
      vectorstore global con k=40 (más contexto).
    - Usa retriever_resumenes para contexto macro (resúmenes diarios).
    """
    data = request.get_json()
    q = data.get("pregunta", "").strip()
    if not q:
        return jsonify({"error": "No se proporcionó una pregunta válida."}), 400

    try:
        # 🧠 1️⃣ Detectar entidades y rango de fechas
        entidades = extraer_entidades(q) if "extraer_entidades" in globals() else {}
        fecha_inicio, fecha_fin, origen_rango = interpretar_rango_fechas(q, df)
        print(f"📅 Rango interpretado para la pregunta: {fecha_inicio} → {fecha_fin} ({origen_rango})")

        tiene_rango = fecha_inicio is not None and fecha_fin is not None

        # 🧠 2️⃣ Filtrar DataFrame por rango ANTES de FAISS (solo si hay rango)
        df_rango = pd.DataFrame()
        if tiene_rango:
            # Asegurarnos de trabajar solo con filas que sí tienen fecha
            df_validas = df.dropna(subset=["Fecha"]).copy()
            df_validas["Fecha_date"] = pd.to_datetime(df_validas["Fecha"], errors="coerce").dt.date

            mask = (df_validas["Fecha_date"] >= fecha_inicio) & (df_validas["Fecha_date"] <= fecha_fin)
            df_rango = df_validas[mask].copy()

            print(f"🧾 Noticias en rango {fecha_inicio} → {fecha_fin}: {len(df_rango)} filas")

        # 🧠 3️⃣ Recuperar resúmenes relevantes (contexto macro)
        resumen_docs = []
        if retriever_resumenes is not None:
            try:
                # Compatibilidad con distintas versiones de LangChain:
                if hasattr(retriever_resumenes, "get_relevant_documents"):
                    resumen_docs = retriever_resumenes.get_relevant_documents(q)
                else:
                    resumen_docs = retriever_resumenes.invoke(q)
            except Exception as e:
                print(f"⚠️ Error al recuperar resúmenes con LangChain: {e}")
                resumen_docs = []
        else:
            print("⚠️ retriever_resumenes es None (aún no hay resúmenes indexados).")

        # Filtrar resúmenes por rango (con fallback si deja todo vacío)
        resumen_docs_filtrados, _ = filtrar_docs_por_rango(resumen_docs, fecha_inicio, fecha_fin)

        bloques_resumen = []
        dias_resumen_usados = []
        for d in resumen_docs_filtrados:
            texto = d.page_content.strip()
            if len(texto) > 600:
                texto = texto[:600] + "..."
            fecha_meta = d.metadata.get("fecha") if d.metadata else None
            if fecha_meta:
                dias_resumen_usados.append(fecha_meta)
                bloques_resumen.append(f"[Resumen {fecha_meta}]\n{texto}")
            else:
                bloques_resumen.append(f"[Resumen sin fecha]\n{texto}")

        bloque_resumenes = "\n\n".join(bloques_resumen) if bloques_resumen else "No se encontraron resúmenes relevantes."

        # 🧠 4️⃣ Recuperar noticias relevantes (contexto micro)
        noticias_docs_filtrados = []

        # 4.A) Si hay rango y SÍ hay noticias en el rango → mini-vectorstore temporal
        if tiene_rango and not df_rango.empty:
            print("🧩 Usando mini-vectorstore temporal de noticias dentro del rango solicitado.")

            docs_rango = []
            for _, row in df_rango.iterrows():
                titulo = str(row.get("Título", "")).strip()
                if not titulo:
                    continue

                fecha_val = row.get("Fecha_date") or row.get("Fecha")
                if pd.notnull(fecha_val):
                    try:
                        fecha_str = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                    except Exception:
                        fecha_str = None
                else:
                    fecha_str = None

                metadata = {
                    "fecha": fecha_str,
                    "fuente": row.get("Fuente"),
                    "enlace": row.get("Enlace"),
                    "cobertura": row.get("Cobertura"),
                    "sentimiento": row.get("Sentimiento"),
                    "termino": row.get("Término"),
                    "idioma": row.get("Idioma"),
                }

                docs_rango.append(Document(page_content=titulo, metadata=metadata))

            if docs_rango:
                mini_vs = LCFAISS.from_documents(docs_rango, embeddings)
                mini_ret = mini_vs.as_retriever(search_kwargs={"k": 40})
                try:
                    if hasattr(mini_ret, "get_relevant_documents"):
                        noticias_docs_filtrados = mini_ret.get_relevant_documents(q)
                    else:
                        noticias_docs_filtrados = mini_ret.invoke(q)
                except Exception as e:
                    print(f"⚠️ Error al recuperar noticias con mini-vectorstore: {e}")
                    noticias_docs_filtrados = []
            else:
                print("⚠️ No se construyeron documentos para el mini-vectorstore (rango vacío tras limpieza).")
                noticias_docs_filtrados = []

        # 4.B) Si NO hay rango o NO hay noticias en ese rango → usar vectorstore global (k=40)
        if (not tiene_rango) or (tiene_rango and df_rango.empty):
            if tiene_rango and df_rango.empty:
                print("ℹ️ No hay noticias en el rango pedido; uso vectorstore global como fallback.")
            else:
                print("ℹ️ Pregunta sin fechas claras; uso vectorstore global con k=40.")

            noticias_docs = []
            if 'vectorstore_noticias' in globals() and vectorstore_noticias is not None:
                try:
                    retriever_global = vectorstore_noticias.as_retriever(search_kwargs={"k": 40})
                    if hasattr(retriever_global, "get_relevant_documents"):
                        noticias_docs = retriever_global.get_relevant_documents(q)
                    else:
                        noticias_docs = retriever_global.invoke(q)
                except Exception as e:
                    print(f"⚠️ Error al recuperar noticias con vectorstore global: {e}")
                    noticias_docs = []
            else:
                print("⚠️ vectorstore_noticias es None (no se construyó índice global de noticias).")

            # En este caso NO filtramos por fecha otra vez: o no hay rango, o el rango estaba vacío
            noticias_docs_filtrados = noticias_docs

        # 4.C) Si no hay NADA de contexto (ni resúmenes ni noticias), responde orientando
        if not resumen_docs_filtrados and not noticias_docs_filtrados:
            mensaje = (
                "No encontré noticias claramente relacionadas con tu pregunta en el histórico disponible. "
                "Intenta reformularla, por ejemplo:\n"
                "- Especifica un tema (aranceles, tasas de interés, nearshoring, etc.)\n"
                "- Menciona un país, ciudad o personaje.\n"
                "- Si quieres un periodo, indica las fechas aproximadas."
            )
            return jsonify({
                "respuesta": mensaje,
                "titulares_usados": [],
                "filtros": {
                    "entidades": entidades,
                    "rango": [str(fecha_inicio), str(fecha_fin)] if (fecha_inicio and fecha_fin) else None,
                    "resumenes_usados": [],
                }
            })

        # 🧾 5️⃣ Construir bloque de titulares + lista para el frontend
        lineas_titulares = []
        titulares_usados = []
        vistos = set()

        for d in noticias_docs_filtrados:
            titulo = d.page_content.strip()
            meta = d.metadata or {}
            fuente = (meta.get("fuente") or "Fuente desconocida").strip()
            enlace = (meta.get("enlace") or "").strip()
            fecha_meta = meta.get("fecha") or ""

            clave = (titulo, fuente, enlace, fecha_meta)
            if clave in vistos:
                continue
            vistos.add(clave)

            linea = f"- {titulo} ({fuente}, {fecha_meta})".strip()
            lineas_titulares.append(linea)

            titulares_usados.append({
                "titulo": titulo,
                "medio": fuente,
                "fecha": fecha_meta,
                "enlace": enlace,
            })

        lineas_titulares = lineas_titulares[:10]
        titulares_usados = titulares_usados[:10]

        if lineas_titulares:
            bloque_titulares = "\n".join(lineas_titulares)
        else:
            bloque_titulares = "No se encontraron titulares específicos, solo contexto general de resúmenes."

        # 🧠 6️⃣ Construir texto final para la chain de LangChain
        texto_usuario = f"""{CONTEXTO_POLITICO}

Responde en español, de forma clara, profesional y analítica.
Usa ÚNICAMENTE la información contenida en los resúmenes y titulares listados abajo.
Si hay resúmenes o titulares en inglés, traduce y sintetiza su contenido.

IMPORTANTE:
- Si el bloque de "Titulares relevantes" que verás más abajo contiene al menos una viñeta que empiece con "-", significa que SÍ hay noticias para trabajar.
- Si la pregunta menciona explícitamente a dos actores, países o bloques (por ejemplo, "Trump y México", "Estados Unidos y China"), asegúrate de explicar de forma equilibrada qué ocurre con cada uno y cómo se relacionan entre sí dentro del periodo solicitado.
- En ese caso, está TOTALMENTE PROHIBIDO escribir frases como:
  "no se reportaron noticias específicas", "no hubo noticias", 
  "no se registraron titulares", "no se encontraron noticias sobre X"
  o cualquier variante equivalente.
- Aunque las noticias sean pocas, indirectas o tangenciales respecto a la pregunta,
  debes describir con precisión qué sí se sabe a partir de esos titulares y por qué
  son relevantes (por ejemplo, porque afectan a México de forma indirecta, etc.).

Solo si el bloque de titulares contiene literalmente el texto:
"No se encontraron titulares específicos, solo contexto general de resúmenes."
puedes explicar que no hay noticias puntuales y limitarte al contexto general.

Si el contexto realmente no contiene ningún titular ni resumen relacionado con la pregunta, indícalo explícitamente y no inventes datos.
Si sí hay información parcial, responde de todas formas describiendo lo que se puede afirmar a partir de esos titulares, sin exagerar pero tampoco diciendo que no hay información.
No menciones titulares individuales, es decir, si algún titular menciona una noticia, cuenta la noticia, no el hecho de que hay un titular de algún medio hablando de eso.
Evita frases como “no se dispone de información específica”; en su lugar, explica directamente lo que sí muestran los titulares.
A menos que el contexto esté totalmente vacío, contesta con un mínimo de 150 palabras.

Pregunta del usuario:
{q}

Rango temporal de referencia (si aplica):
{fecha_inicio} → {fecha_fin}

Resúmenes relevantes:
{bloque_resumenes}

Titulares relevantes:
{bloque_titulares}

Respuesta:
"""

        # 🧩 7️⃣ Llamada a LangChain (ChatOpenAI + PromptTemplate)
        texto_respuesta = chain_pregunta.invoke({"texto_usuario": texto_usuario}).strip()

        return jsonify({
            "respuesta": texto_respuesta,
            "titulares_usados": titulares_usados,
            "filtros": {
                "entidades": entidades,
                "rango": [str(fecha_inicio), str(fecha_fin)] if (fecha_inicio and fecha_fin) else None,
                "resumenes_usados": dias_resumen_usados,
            }
        })

    except Exception as e:
        print(f"❌ Error en /pregunta (LangChain): {e}")
        return jsonify({"error": str(e)}), 500




#correoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
@app.route("/enviar_email", methods=["POST"])
def enviar_email():
    data = request.get_json()
    email = data.get("email")
    fecha_str = data.get("fecha")
    fecha_dt = pd.to_datetime(fecha_str).date()

    resultado = generar_resumen_y_datos(fecha_str)
    if "error" in resultado:
        return jsonify({"mensaje": resultado["error"]}), 404

    titulares_info = resultado.get("titulares", [])
    titulares_info_en = resultado.get("titulares_en", [])
    resumen_texto = resultado.get("resumen", "")
        # 🔹 Convertir saltos de línea en HTML para conservar párrafos en el correo
    resumen_html = (resumen_texto or "").replace("\n\n", "<br><br>").replace("\n", "<br>")


    if not resumen_texto:
        archivo_resumen = os.path.join("resumenes", f"resumen_{fecha_str}.txt")
        if os.path.exists(archivo_resumen):
            with open(archivo_resumen, "r", encoding="utf-8") as f:
                resumen_texto = f.read()

    # ☁️ Nube
    archivo_nube = os.path.join("nubes", f"nube_{fecha_str}.png")

        # 📊 Indicadores económicos
    # 📊 Indicadores económicos
# 📊 Buscar datos económicos más recientes disponibles
    df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date

    # Intentar primero la fecha exacta
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt]

    # Si no hay datos para ese día, usar la fecha más cercana anterior
    if economia_dia.empty:
        ultima_fecha = df_economia[df_economia["Fecha"] <= fecha_dt]["Fecha"].max()
        if pd.notnull(ultima_fecha):
            economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]

    # Si sigue vacío (p. ej., todos los datos son posteriores), usar la más reciente disponible
    if economia_dia.empty and not df_economia.empty:
        ultima_fecha = df_economia["Fecha"].max()
        economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]

    # 🧩 Fallback adicional a nivel columna: usar último valor válido anterior
    import numpy as np
    for col in ORDEN_COLUMNAS:
        if col in economia_dia.columns and (
            economia_dia.iloc[0][col] in ["", None, np.nan] or pd.isna(economia_dia.iloc[0][col])
        ):
            valores_previos = df_economia[df_economia["Fecha"] <= fecha_dt][col].dropna()
            if not valores_previos.empty:
                economia_dia.loc[economia_dia.index[0], col] = valores_previos.iloc[-1]


    print(f"📅 Fecha económica usada para correo: {economia_dia['Fecha'].iloc[0] if not economia_dia.empty else 'Sin datos'}")


    if not economia_dia.empty:
        economia_dia = economia_dia.copy()

        # 🔹 Tipo de cambio
        for col in ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")


        # 🔹 Tasas
        for col in ["Tasa de Interés Objetivo Banxico", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días",
                    "Tasa efectiva FED", "Rango objetivo superior FED", "Rango objetivo inferior FED"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(formatear_porcentaje_decimal)

        # 🔹 SOFR
        if "SOFR" in economia_dia.columns:
            economia_dia["SOFR"] = economia_dia["SOFR"].apply(format_porcentaje_directo)

        # 🔹 Bolsas
        for col in ["% Dow Jones", "% S&P500", "% Nasdaq"]:
            if col in economia_dia.columns:
                economia_dia[col] = economia_dia[col].apply(format_signed_pct)

        # 🔹 Inflación MEX: usar df_infl_mx directo
        for col in ["Inflación Anual MEX", "Inflación Subyacente MEX"]:
            if col in df_infl_mx.columns:
                valores_previos = df_infl_mx[df_infl_mx["Fecha"] <= fecha_dt][col].dropna()
                if not valores_previos.empty:
                    ultimo_valor = pd.to_numeric(valores_previos.iloc[-1], errors="coerce")
                    economia_dia[col] = f"{ultimo_valor*100:.2f}%" if pd.notnull(ultimo_valor) else ""

        # 🔹 Inflación US: usar df_infl_us directo (igual que MEX)
        for col in ["Inflación Anual US", "Inflación Subyacente US"]:
            if col in df_infl_us.columns:
                valores_previos = df_infl_us[df_infl_us["Fecha"] <= fecha_dt][col].dropna()
                if not valores_previos.empty:
                    ultimo_valor = pd.to_numeric(valores_previos.iloc[-1], errors="coerce")
                    economia_dia[col] = f"{ultimo_valor*100:.2f}%" if pd.notnull(ultimo_valor) else ""




        # Ordenar columnas
        economia_dia = economia_dia.reindex(columns=ORDEN_COLUMNAS)
        economia_dict = OrderedDict()
        for col in ORDEN_COLUMNAS:
            economia_dict[col] = economia_dia.iloc[0][col]

        # 🔹 Construcción manual en filas
        filas = [
            ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"],
            ["Tasa de Interés Objetivo Banxico", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días"],
            ["Tasa efectiva FED", "Rango objetivo superior FED", "Rango objetivo inferior FED"],
            ["SOFR", "% Dow Jones", "% S&P500", "% Nasdaq"],
            ["Inflación Anual MEX", "Inflación Subyacente MEX",
             "Inflación Anual US", "Inflación Subyacente US"]
        ]

        indicadores_html = ""
        for fila in filas:
            indicadores_html += "<div style='display:flex; flex-wrap:wrap; gap:12px; margin-top:10px;'>"
            for col in fila:
                valor = economia_dict.get(col, "")
                indicadores_html += f"""
                <div style="flex:1 1 calc(25% - 12px); background:#fff; border:1px solid #ddd; border-radius:12px; padding:12px; text-align:center; min-width:150px;">
                    <div style="font-size:0.85rem; color:#7D7B78; margin-bottom:6px;">{col}</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#111;">{valor}</div>
                </div>
                """
            indicadores_html += "</div>"
    else:
        indicadores_html = "<p>No hay datos económicos</p>"

    # ---- CONFIGURACIÓN DEL CORREO ----
    remitente = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_PASS")

    destinatario = email

    msg = MIMEMultipart()
    msg["From"] = formataddr(("Monitoreo +", remitente))  # 👈 nombre visible
    msg["To"] = destinatario
    msg["Subject"] = f"Resumen de noticias {fecha_str}"


    # 📧 Plantilla HTML con estilo
    cuerpo = f"""
    
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center" style="width:100%; max-width:800px; font-family:Montserrat,Arial,sans-serif; border-collapse:collapse; margin:auto;">
    <!-- Header con fondo blanco -->
    <tr>
        <td style="background:#fff; padding:16px 20px; border-bottom:2px solid #e5e7eb;">
        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
            <tr>
                <td align="left" style="vertical-align:middle;">
                   <img src="cid:logo"
                        alt="Industrial Gate"
                        width="180"
                        style="
                            display:block;
                            margin:0 auto;
                            width:180px;
                            max-width:180px;
                            height:auto;
                            -ms-interpolation-mode:bicubic;
                        "> 
                </td>
                <td align="right" style="font-weight:700; font-size:1.2rem; color:#111;">
                    Monitoreo<span style="color:#FFB429;">+</span>
                </td>
            </tr>
        </table>
        </td>
    </tr>

    <!-- Bloque gris con contenido -->
    <tr>
        <td style="background:#f9f9f9; padding:20px; border:1px solid #e5e7eb; border-radius:0 0 12px 12px;">
        
        <!-- Resumen -->
        <h2 style="font-size:1.4rem; font-weight:700; margin-bottom:14px; color:#111;">
            📅 Resumen diario de noticias — {fecha_str}
        </h2>
        <div style="background:#fff; border:1px solid #ddd; border-radius:12px; padding:20px; margin-bottom:20px;">
            <p style="color:#555; line-height:1.7; text-align:justify;">{resumen_html}</p>
        </div>

        <!-- Indicadores -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">📊 Indicadores económicos</h3>
        {indicadores_html}

        <!-- Titulares español -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">🗞️ Principales titulares en español</h3>
        <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
            {''.join([f"<div style='padding:10px; border:1px solid #ddd; border-radius:12px; background:#fff; max-width:100%; word-break:normal; white-space:normal; overflow-wrap:anywhere;'><a href='{t['enlace']}' style='color:#0B57D0; font-weight:600; text-decoration:none;'>{t['titulo']}</a><br><small style='color:#7D7B78;'>• {t['medio']}</small></div>" for t in titulares_info if t.get('idioma','es')=='es'])}
        </div>

        <!-- Titulares inglés -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">🗞️ Principales titulares en inglés</h3>
        <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
            {''.join([f"<div style='padding:10px; border:1px solid #ddd; border-radius:12px; background:#fff; max-width:100%; word-break:normal; white-space:normal; overflow-wrap:anywhere;'><a href='{t['enlace']}' style='color:#0B57D0; font-weight:600; text-decoration:none;'>{t['titulo']}</a><br><small style='color:#7D7B78;'>• {t['medio']}</small></div>" for t in titulares_info_en])}
        </div>

        <!-- Nube -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">☁️ Nube de palabras</h3>
        <div style="text-align:center; margin-top:12px;">
            <img src="cid:nube" alt="Nube de palabras" style="width:100%; max-width:600px; border-radius:12px; border:1px solid #ddd;" />
        </div>

        </td>
    </tr>
    </table>
    """


    msg.attach(MIMEText(cuerpo, "html"))

    # 📎 Adjuntar nube inline
    if os.path.exists(archivo_nube):
        with open(archivo_nube, "rb") as img_file:
            imagen = MIMEImage(img_file.read())
            imagen.add_header("Content-ID", "<nube>")
            imagen.add_header("Content-Disposition", "inline", filename=archivo_nube)
            msg.attach(imagen)

    # 📎 Adjuntar logo del cliente inline
    if os.path.exists("logo.png"):  # asegúrate de poner el logo en tu carpeta del proyecto
        with open("logo.png", "rb") as logo_file:
            logo = MIMEImage(logo_file.read())
            logo.add_header("Content-ID", "<logo>")
            logo.add_header("Content-Disposition", "inline", filename="logo.png")
            msg.attach(logo)        

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Gmail
        server.starttls()
        server.login(remitente, password)
        server.sendmail(remitente, destinatario, msg.as_string())  # 👈 enviar
        server.quit()
        return jsonify({"mensaje": f"✅ Correo enviado a {destinatario}"})
    
    except Exception as e:
    
        return jsonify({"mensaje": f"❌ Error al enviar correo: {e}"})



@app.route("/nube/<filename>")
def serve_nube(filename):
    return send_from_directory("nubes", filename)

@app.route("/fechas", methods=["GET"])
def fechas():
    global df
    try:
        if df.empty:
            print("⚠️ DataFrame vacío al solicitar /fechas")
            return jsonify([])

        # Normalizar tipo de dato (maneja tanto datetime64 como date)
        if pd.api.types.is_datetime64_any_dtype(df["Fecha"]):
            fechas_unicas = df["Fecha"].dropna().dt.date.unique()
        else:
            # Si ya son objetos date o strings convertibles
            fechas_unicas = pd.to_datetime(df["Fecha"], errors="coerce").dropna().dt.date.unique()

        fechas_ordenadas = sorted(fechas_unicas, reverse=True)
        fechas_str = [f.strftime("%Y-%m-%d") for f in fechas_ordenadas]

        print(f"🗓️ /fechas → {len(fechas_str)} fechas detectadas (rango {fechas_str[-1]} → {fechas_str[0]})")
        return jsonify(fechas_str)

    except Exception as e:
        print(f"❌ Error en /fechas: {e}")
        return jsonify([])




# ------------------------------
# 📑 Endpoint para análisis semanal
# ------------------------------
@app.route("/reporte_semanal", methods=["GET"]) 
def reporte_semanal():
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reporte_semanal")
    os.makedirs(carpeta, exist_ok=True)

    archivos = [
        f for f in os.listdir(carpeta)
        if f.lower().endswith(".pdf")
    ]
    archivos.sort(reverse=True)  # más recientes primero

    resultados = []
    for f in archivos:
        # Extraer fechas del nombre (ej: analisis_2025-08-25_a_2025-08-29.pdf)
        match = re.search(r"(\d{4}-\d{2}-\d{2})_a_(\d{4}-\d{2}-\d{2})", f)
        if match:
            fecha_inicio = datetime.strptime(match.group(1), "%Y-%m-%d")
            fecha_fin = datetime.strptime(match.group(2), "%Y-%m-%d")
            nombre_bonito = f"Reporte semanal: {fecha_inicio.day}–{fecha_fin.day} {nombre_mes(fecha_fin)}"
        else:
            nombre_bonito = f  # fallback al nombre del archivo

        resultados.append({
            "nombre": nombre_bonito,
            "url": f"/reporte/{f}"
        })

    return jsonify(resultados)

@app.route("/reporte/<path:filename>", methods=["GET"])
def descargar_reporte(filename):
    return send_from_directory("reporte_semanal", filename, as_attachment=False)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
