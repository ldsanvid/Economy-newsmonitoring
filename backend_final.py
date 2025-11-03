from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dateparser
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

import base64
import requests
import calendar
from babel.dates import format_date
from flask_cors import CORS
from s3_utils import s3_download_all as r2_download_all, s3_upload as r2_upload
import numpy as np
import faiss



def nombre_mes(fecha):
    """Devuelve la fecha con mes en español, ej: 'agosto 2025'"""
    return format_date(fecha, "LLLL yyyy", locale="es").capitalize()


# ------------------------------
# 🔑 Configuración API y Flask
# ------------------------------
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
df_economia = df_economia.merge(df_tasas_us, on=["Año", "Fecha"], how="outer").fillna("")

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

# ------------------------------
# 🧠 Carga del índice FAISS (para búsqueda semántica) — con rutas absolutas y diagnóstico
# ------------------------------
import faiss

USE_FAISS = True
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_dir = os.path.join(base_dir, "faiss_index")

    index_path = os.path.join(faiss_dir, "noticias_index.faiss")
    meta_path = os.path.join(faiss_dir, "noticias_metadata.csv")

    # 🔍 Verifica existencia antes de cargar
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No se encontró el archivo FAISS: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {meta_path}")

    index = faiss.read_index(index_path)
    df_metadata = pd.read_csv(meta_path)

    # Validar columnas necesarias
    columnas_requeridas = ["id","Fecha","Título","Fuente","Enlace","Cobertura","Término","Sentimiento"]
    if not all(col in df_metadata.columns for col in columnas_requeridas):
        raise ValueError("❌ El CSV de metadatos FAISS no contiene todas las columnas requeridas.")

    print(f"✅ FAISS cargado correctamente con {len(df_metadata)} registros.")
    print(f"📂 Ruta del índice: {index_path}")

except Exception as e:
    USE_FAISS = False
    print(f"⚠️ No se pudo cargar FAISS, se usará TF-IDF. Motivo: {e}")


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
- PROFEPA es la Procuraduría Federal de Protección al Ambiente de México.
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

# ------------------------------
# 🔍 Búsqueda semántica con FAISS o TF-IDF (según disponibilidad)
# ------------------------------
def buscar_semantica_noticias(query, df_base, top_k=200):
    """
    Si hay FAISS: busca en TODO el corpus y luego intersecta con df_base (filtros de fecha/entidades/sentimiento).
    Si no hay FAISS: hace TF-IDF sobre df_base.
    Devuelve un DataFrame con las top coincidencias (máx 5).
    """
    q = query.strip()
    if not USE_FAISS:
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df_base["Título"])
        v = tfidf.transform([q])
        sims = cosine_similarity(v, X).flatten()
        idx = sims.argsort()[-5:][::-1]
        return df_base.iloc[idx]

    try:
        emb_q = client.embeddings.create(
            model="text-embedding-3-small", input=q
        ).data[0].embedding

        vq = np.array(emb_q, dtype="float32")[np.newaxis, :]
        sims, ids = index.search(vq, top_k)
        candidatos = df_metadata.iloc[ids[0]].copy()

        # Intersección con df_base por campos clave
        clave = ["Título","Fuente","Enlace"]
        mergeado = candidatos.merge(
            df_base[clave + ["Fecha","Cobertura","Término","Sentimiento"]],
            on=clave, how="inner"
        )

        if mergeado.empty:
            tfidf = TfidfVectorizer()
            X = tfidf.fit_transform(df_base["Título"])
            v = tfidf.transform([q])
            sims = cosine_similarity(v, X).flatten()
            idx = sims.argsort()[-5:][::-1]
            return df_base.iloc[idx]

        orden = {t:i for i,t in enumerate(candidatos["Título"].tolist())}
        mergeado["__rank"] = mergeado["Título"].map(orden)
        mergeado = mergeado.sort_values("__rank").drop(columns="__rank")
        return mergeado.head(10)

    except Exception:
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df_base["Título"])
        v = tfidf.transform([q])
        sims = cosine_similarity(v, X).flatten()
        idx = sims.argsort()[-5:][::-1]
        return df_base.iloc[idx[:10]]



# 6️⃣ Seleccionar titulares más relevantes (TF-IDF + coseno)
def seleccionar_titulares_relevantes(titulares, pregunta):
    if not titulares:
        return []
    vectorizer = TfidfVectorizer().fit(titulares + [pregunta])
    vectores = vectorizer.transform(titulares + [pregunta])
    similitudes = cosine_similarity(vectores[-1], vectores[:-1]).flatten()
    indices_similares = similitudes.argsort()[-5:][::-1]
    return [titulares[i] for i in indices_similares]

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
                ultimos = df_prev.tail(5)
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

Redacta un resumen de noticias del {fecha_str} dividido en cinco párrafos. Tono profesional, objetivo y dirigido a tomadores de decisiones. De 400 palabras. Antes de empezar a redactar, revisa {CONTEXTO_ANTERIOR}, si existe. Utilízalo para darle continuidad narrativa. Si en el resumen anterior se presenta x noticia y en las noticias del nuevo día se vuelve a hacer mención de la misma, retómalo para únicamente contar lo que es nuevo (ya sea nuevos desarrollos, nuevas reacciones) pero sin volver a contarlo como si fuera la primera vez. 

Luego de hacer esa revisión y considerarla antes de empezar a redactar:

Primer párrafo: Describe y contextualiza el tema más repetido del día (qué, quién, cómo). Si esta noticia ya fue mencionada en días previos, menciona que es una continuación o extensión de una noticia ya ocurrida y no repitas los mismos detalles, es decir, da información nueva. En caso de que sea una noticia nueva, sin dar tu punto de vista, quiero que presentes todo lo que encontraste sobre ese tema (declaraciones de diversos actores u organismos, por ejemplo, para mostrar todas las versiones).

Segundo párrafo: Sin repetir la noticia sobre la que te enfocaste en el primer párrafo, en este párrafo quiero que resumas las noticias, tanto de cobertura nacional como internacional, que sean sobre aranceles, tasas de interés, acuerdo comercial, banco central y reserva federal. Permítete hacer este párrafo más extenso que el resto si es necesario. Si esta noticia ya fue mencionada en días previos, menciona que es una continuación o extensión de una noticia ya ocurrida y no repitas los mismos detalles, es decir, da información nueva

Tercer párrafo: Resume brevemente las noticias que son de cobertura de algún estado de México (locales), excluyendo aquellas de cobertura nacional o internacional SIN REPETIR NOTICIAS DE LOS PÁRRAFOS O DÍAS PREVIOS. Menciona el estado o ciudad de cobertura de cada noticia. No repitas noticias mencionadas en los párrafos anteriores ni inventes cosas. Reserva todo lo relativo a fibras, naves industriales, parques industriales, hub logístico, hub industrial, real estate industrial, sector industrial o sector mobiliario industrial para el último párrafo. Usa este párrafo para cubrir noticias relacionadas con industria automotriz, transporte, construcción, fusiones y cadenas de suministro.

Cuarto párrafo: Por último, resume de forma general las noticias que tienen que ver con fibras, naves industriales, parques industriales, hub logístico, hub industrial, real estate industrial, sector industrial o sector mobiliario industrial SIN REPETIR ALGUNA NOTICIAS MENCIONADA EN PÁRRAFOS O DÍAS PREVIOS. Recuerda, temas no arancelarios. Empieza diciendo "finalmente en otros temas económicos", sin recalcar de que se trata de noticias del ámbito local o nacional.

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
    resumen_file = f"resumen_{fecha_str}.txt"
    if os.path.exists(resumen_file):
        with open(resumen_file, "r", encoding="utf-8") as f:
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
        with open(resumen_file, "w", encoding="utf-8") as f:
            f.write(resumen_texto)

    # --- Generar nube ---
    os.makedirs("nubes", exist_ok=True)
    archivo_nube = f"nube_{fecha_str}.png"
    archivo_nube_path = os.path.join("nubes", archivo_nube)
    generar_nube(noticias_dia["Título"].tolist(), archivo_nube_path)

    # 📊 Indicadores económicos
    df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date

    # Filtrar datos económicos por fecha
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt]

    # Si no hay datos exactos, usar el más reciente antes de esa fecha
    if economia_dia.empty:
        ultima_fecha = df_economia[df_economia["Fecha"] <= fecha_dt]["Fecha"].max()
        economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]

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

        # 🔹 Inflación US: se queda leyendo de df_economia
        for col in ["Inflación Anual US", "Inflación Subyacente US"]:
            if col in df_economia.columns:
                valores_previos = df_economia[df_economia["Fecha"] <= fecha_dt][col].dropna()
                if not valores_previos.empty:
                    ultimo_valor = pd.to_numeric(valores_previos.iloc[-1], errors="coerce")
                    economia_dia[col] = f"{ultimo_valor*100:.2f}%" if pd.notnull(ultimo_valor) else ""



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



    # 📰 Titulares sin repetir medios
    titulares_info = []
    usados_medios = set()

    def agregar_titulares(df_origen, max_count, idioma_filtrado="es"):
        added = 0
        for _, row in df_origen.iterrows():
            medio = row["Fuente"]
            idioma = str(row.get("Idioma", "es")).strip().lower()
            if idioma in ["en", "ingles", "inglés"]:
                idioma = "en"
            else:
                idioma = "es"

            if idioma != idioma_filtrado:
                continue

            if medio not in usados_medios:
                titulares_info.append({
                    "titulo": row["Título"],
                    "medio": medio,
                    "enlace": row["Enlace"],
                    "idioma": idioma
                })
                usados_medios.add(medio)
                added += 1
            if added >= max_count:
                break



    # 2 nacionales + 2 locales + 2 internacionales + 2 otros = 8 titulares distintos
    agregar_titulares(noticias_nacionales, 2, idioma_filtrado="es")
    agregar_titulares(noticias_locales, 2, idioma_filtrado="es")
    agregar_titulares(noticias_internacionales_forzadas, 2, idioma_filtrado="es")
    agregar_titulares(noticias_otras_forzadas, 2, idioma_filtrado="es")


    # 📰 Titulares en inglés (máx. 8)
    titulares_info_en = []
    if "Idioma" in noticias_dia.columns:
        notas_en = noticias_dia[noticias_dia["Idioma"].str.lower().isin(["en", "inglés", "ingles"])]
        notas_en = notas_en.dropna(subset=["Título"]).drop_duplicates(subset=["Título", "Fuente", "Enlace"])
        usados_medios_en = set()
        for _, row in notas_en.iterrows():
            medio = row["Fuente"]
            if medio not in usados_medios_en:
                titulares_info_en.append({
                    "titulo": row["Título"],
                    "medio": medio,
                    "enlace": row["Enlace"],
                    "idioma": "en"
                })
                usados_medios_en.add(medio)
            if len(titulares_info_en) >= 8:
                break

        # ------------------------------
    # 💾 Guardar resumen y subir a S3
    # ------------------------------
    # ------------------------------
    # 💾 Guardar resumen y subir a S3 (modo append con control de duplicados)
    # ------------------------------
    try:
        from s3_utils import s3_upload

        os.makedirs("faiss_index", exist_ok=True)
        resumen_meta_path = "faiss_index/resumenes_metadata.csv"

        df_resumen = pd.DataFrame([{
            "fecha": str(fecha_dt),
            "archivo_txt": f"resumen_{fecha_str}.txt",
            "nube": archivo_nube,
            "titulares": len(titulares_info),
            "resumen": resumen_texto.strip().replace("\n", " ")
        }])

        # Si ya existe el archivo, lo leemos y agregamos (sin duplicar fechas)
        if os.path.exists(resumen_meta_path):
            df_prev = pd.read_csv(resumen_meta_path)

            if str(fecha_dt) not in df_prev["fecha"].astype(str).values:
                df_total = pd.concat([df_prev, df_resumen], ignore_index=True)
                print(f"🆕 Agregado nuevo resumen para {fecha_dt}")
            else:
                print(f"♻️ Reemplazando resumen existente para {fecha_dt}")
                df_prev.loc[df_prev["fecha"].astype(str) == str(fecha_dt), :] = df_resumen.iloc[0]
                df_total = df_prev
        else:
            df_total = df_resumen

        # Guardar y subir
        df_total.to_csv(resumen_meta_path, index=False, encoding="utf-8")
        print(f"💾 Guardado local de resumenes_metadata.csv con {len(df_total)} fila(s) totales")
        s3_upload("resumenes_metadata.csv")
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
        from s3_utils import s3_upload
        s3_upload("resumenes_index.faiss")
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
    data = request.get_json()
    fecha_str = data.get("fecha")
    if not fecha_str:
        return jsonify({"error": "Debe especificar una fecha"}), 400

    resultado = generar_resumen_y_datos(fecha_str)

    if "error" in resultado:
        return jsonify(resultado), 404

    return jsonify(resultado)

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

#pregunta!!!!    
# ------------------------------
# 🤖 Endpoint /pregunta (ahora con RAG real + FAISS)
# ------------------------------
@app.route("/pregunta", methods=["POST"])
def pregunta():
    data = request.get_json()
    q = data.get("pregunta", "").strip()
    if not q:
        return jsonify({"error": "No se proporcionó una pregunta válida."}), 400

    try:
        # 🧠 1️⃣ Detectar fechas (una o rango)
        fechas_detectadas = list(dateparser.search.search_dates(q, languages=['es', 'en']))
        fecha_inicio = fecha_fin = None

        if len(fechas_detectadas) == 1:
            fecha_inicio = fecha_fin = fechas_detectadas[0][1].date()
        elif len(fechas_detectadas) >= 2:
            fecha_inicio = fechas_detectadas[0][1].date()
            fecha_fin = fechas_detectadas[1][1].date()

        # 🧩 2️⃣ Detectar entidades clave
        entidades = extraer_entidades(q)

        # 🧩 3️⃣ Filtrado del DataFrame principal por fecha y entidades
        df_filtrado = df.copy()
        if fecha_inicio and fecha_fin:
            df_filtrado = df_filtrado[
                (pd.to_datetime(df_filtrado["Fecha"]).dt.date >= fecha_inicio)
                & (pd.to_datetime(df_filtrado["Fecha"]).dt.date <= fecha_fin)
            ]
        elif fecha_inicio:
            df_filtrado = df_filtrado[
                pd.to_datetime(df_filtrado["Fecha"]).dt.date == fecha_inicio
            ]

        if not df_filtrado.empty:
            df_filtrado = filtrar_titulares(df_filtrado, entidades, detectar_sentimiento_deseado(q))

        if df_filtrado.empty:
            return jsonify({
                "respuesta": f"No encontré noticias relacionadas con tu pregunta: '{q}'.",
                "titulares_usados": []
            })

        # 🧠 4️⃣ Búsqueda semántica con FAISS (o TF-IDF fallback)
        resultados = buscar_semantica_noticias(q, df_filtrado, top_k=200)
        if resultados.empty:
            return jsonify({
                "respuesta": f"No encontré coincidencias semánticas para '{q}'.",
                "titulares_usados": []
            })

        top_noticias = resultados.head(10)

        # 🧩 5️⃣ Construcción del contexto
        contexto = "\n".join([
            f"- {row['Título']} ({row['Fuente']})"
            for _, row in top_noticias.iterrows()
        ])

        # 🧠 6️⃣ Prompt para GPT
        prompt = f"""{CONTEXTO_POLITICO}

Responde la siguiente pregunta de forma clara, profesional y analítica,
usando únicamente los titulares listados a continuación.
No inventes datos; si la información no está presente, indícalo.

Pregunta: {q}

Titulares relevantes:
{contexto}

Respuesta:
"""

        # 🧩 7️⃣ Llamada a OpenAI
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un analista de medios experto en política y economía. Responde solo con base en los titulares dados."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=700
        )

        texto_respuesta = respuesta.choices[0].message.content.strip()

        titulares_usados = [
            {
                "titulo": row["Título"],
                "medio": row["Fuente"],
                "fecha": str(row["Fecha"]),
                "enlace": row["Enlace"]
            }
            for _, row in top_noticias.iterrows()
        ]

        return jsonify({
            "respuesta": texto_respuesta,
            "titulares_usados": titulares_usados,
            "filtros": {
                "entidades": entidades,
                "rango": [str(fecha_inicio), str(fecha_fin)] if fecha_inicio else None
            }
        })

    except Exception as e:
        print(f"❌ Error en /pregunta: {e}")
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

    if not resumen_texto:
        archivo_resumen = os.path.join("resumenes", f"resumen_{fecha_str}.txt")
        if os.path.exists(archivo_resumen):
            with open(archivo_resumen, "r", encoding="utf-8") as f:
                resumen_texto = f.read()

    # ☁️ Nube
    archivo_nube = os.path.join("nubes", f"nube_{fecha_str}.png")

        # 📊 Indicadores económicos
    # 📊 Indicadores económicos
    df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date

    # Filtrar datos económicos por fecha
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt]

    # Si no hay datos exactos, usar el más reciente antes de esa fecha
    if economia_dia.empty:
        ultima_fecha = df_economia[df_economia["Fecha"] <= fecha_dt]["Fecha"].max()
        economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]

    if not economia_dia.empty:
        economia_dia = economia_dia.copy()

        # 🔹 Tipo de cambio
        for col in ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(formatear_porcentaje)

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

        # 🔹 Inflación US: se queda leyendo de df_economia
        for col in ["Inflación Anual US", "Inflación Subyacente US"]:
            if col in df_economia.columns:
                valores_previos = df_economia[df_economia["Fecha"] <= fecha_dt][col].dropna()
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
            <td align="left">
                <img src="cid:logo" alt="Cliente" style="height:40px;">
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
            <p style="color:#555; line-height:1.7; text-align:justify;">{resumen_texto}</p>
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
