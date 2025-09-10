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
from collections import OrderedDict
from flask import send_file
from collections import OrderedDict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import base64
import requests
import calendar
from babel.dates import format_date

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
@app.route("/")
def home():
    return send_file("index.html")
# ------------------------------
# 📂 Carga única de datos
# ------------------------------
# Noticias
try:
    df = pd.read_csv("noticias_fondo con todas las fuentes_rango_03-07-2025.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("noticias_fondo con todas las fuentes_rango_03-07-2025.csv", encoding="latin-1")
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Fecha", "Título"])

# Indicadores económicos
df_tipo_cambio = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="Tipo de Cambio")
df_tasas = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="Tasas de interés")
df_economia = pd.merge(df_tipo_cambio, df_tasas, on=["Año", "Fecha"], how="outer").fillna("")
# Cargar hojas adicionales
df_sofr = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="Treasuries_SOFR")
df_wall = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="Wallstreet")
df_infl_us = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="InflaciónUS")
df_infl_us = df_infl_us.rename(columns={
    "Inflación Anual": "Inflación Anual US",
    "Inflación Subyacente": "Inflación Subyacente US"
})
df_infl_mx = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="InflaciónMEX")
df_infl_mx = df_infl_mx.rename(columns={
    "Inflación Anual": "Inflación Anual MEX",
    "Inflación Subyacente": "Inflación Subyacente MEX"
})

# Unificar fechas
for df_tmp in [df_sofr, df_wall, df_infl_us, df_infl_mx]:
    df_tmp["Fecha"] = pd.to_datetime(df_tmp["Fecha"], errors="coerce").dt.date

# Unir con df_economia
df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date
df_economia = df_economia.merge(df_sofr[["Fecha", "SOFR"]], on="Fecha", how="left")
df_economia = df_economia.merge(
    df_infl_us[["Fecha", "Inflación Anual US", "Inflación Subyacente US"]],
    on="Fecha", how="left"
)
df_economia = df_economia.merge(
    df_infl_mx[["Fecha", "Inflación Anual MEX", "Inflación Subyacente MEX"]],
    on="Fecha", how="left"
)
df_economia = df_economia.merge(df_wall[["Fecha", "% Dow Jones", "% S&P500", "% Nasdaq"]], on="Fecha", how="left")

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
- Verónica Delgadillo es la actual Alcaldesa de Guadalajara 
"""
# 1️⃣ Extraer fecha desde texto
def extraer_fecha(pregunta):
    posibles = re.findall(r"\d{1,2} de [a-zA-Z]+(?: de \d{4})?", pregunta)
    if posibles:
        fecha = dateparser.parse(posibles[0], languages=['es'])
        return fecha.date() if fecha else None
    return None

# 2️⃣ Obtener fecha más reciente disponible
def obtener_fecha_mas_reciente(df):
    return df["Fecha"].max().date()

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
        "Fed": ["Federal Reserve"]

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
        return "rgb(0, 0, 0)"
    elif font_size >= 40:
        return "rgb(255, 180, 41)"
    else:
        return "rgb(125, 123, 120)"

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

    prompt = f"""
    {CONTEXTO_POLITICO}

Redacta un resumen de noticias del {fecha_str} dividido en cuatro párrafos. Tono profesional, objetivo y dirigido a tomadores de decisiones. Máximo 250 palabras.

Primer párrafo: Describe y contextualiza el tema más repetido del día (qué, quién, cómo).

Segundo párrafo: Si el tema más repetido del día es de noticias nacionales, usa este segundo párrafo para profundizar en el segundo tema más importante que sea de noticias internacionales. Si en cambio el tema más repetido del día no es de noticias nacionales, este segundo párrafo debe enfocarse en el tema más relevante de noticias nacionales.

Tercer párrafo: Resume brevemente las noticias que son de cobertura de algún estado de México (locales), excluyendo aquellas de cobertura nacional o internacional. Menciona el estado o ciudad de cobertura de cada noticia. No repitas noticias mencionadas en los párrafos anteriores ni inventes cosas. Reserva todo lo relativo a fibras, naves industriales, parques industriales, hub logístico, hub industrial, real estate industrial, sector industrial o sector mobiliario industrial.

Cuarto párrafo: Por último, resume de forma general las noticias que no están relacionadas con aranceles y que tienen que ver con fibras, naves industriales, parques industriales, hub logístico, hub industrial, real estate industrial, sector industrial o sector mobiliario industrial SIN REPETIR ALGUNA NOTICIAS MENCIONADA EN PÁRRAFOS PREVIOS. Evita repetir noticias mencionadas en los otros párrafos ni inventes cosas. Recuerda, temas no arancelarios. Empieza diciendo "finalmente en otros temas económicos", sin recalcar de que se trata de noticias del ámbito local o nacional..

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
    # Filtrar datos económicos
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt]
    # Si la inflación USA está vacía en el día seleccionado, usar el valor más reciente disponible
    for col in ["Inflación Anual MEX", "Inflación Subyacente MEX",
            "Inflación Anual US", "Inflación Subyacente US"]:
        if col in economia_dia.columns:
            economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
            economia_dia[col] = economia_dia[col].apply(formatear_porcentaje)

    # Si no hay datos exactos, usar el más reciente antes de esa fecha
    if economia_dia.empty:
        ultima_fecha = df_economia[df_economia["Fecha"] <= fecha_dt]["Fecha"].max()
        economia_dia = df_economia[df_economia["Fecha"] == ultima_fecha]

    if economia_dia.empty:
        economia_dict = {}
    else:
        economia_dia = economia_dia.copy()

        # Convertir a numérico antes de formatear
        for col in ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")

        for col in ["Tasa de Interés Objetivo", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")
        
        # Inflaciones (formatear siempre con %)
        def formatear_porcentaje(x):
            if pd.isnull(x):
                return ""
            # Si es decimal (0.034 → 3.4%) o ya viene como 3.4 (→ 3.4%)
            return f"{(x*100 if abs(x) <= 1 else x):.2f}%"

        for col in ["Inflación Anual MEX", "Inflación Subyacente MEX",
                    "Inflación Anual US", "Inflación Subyacente US"]:
            if col in economia_dia.columns:
                economia_dia[col] = pd.to_numeric(economia_dia[col], errors="coerce")
                economia_dia[col] = economia_dia[col].apply(formatear_porcentaje)

        # Reordenar columnas según el orden deseado
        # Reordenar columnas y agregar nuevos indicadores
        orden_columnas = [
            "Tipo de Cambio FIX",
            "Nivel máximo",
            "Nivel mínimo",
            "Tasa de Interés Objetivo",
            "TIIE 28 días",
            "TIIE 91 días",
            "TIIE 182 días",
            "SOFR",
            "% Dow Jones",
            "% S&P500",
            "% Nasdaq",
            "Inflación Anual MEX",
            "Inflación Subyacente MEX",
            "Inflación Anual US",
            "Inflación Subyacente US"
        ]
        def format_porcentaje_directo(x):
            try:
                x_clean = str(x).replace('%','').strip()
                return f"{float(x_clean)*100:.2f}%"
            except:
                return ""
        # Formato para nuevos indicadores
        economia_dia["SOFR"] = economia_dia["SOFR"].apply(format_porcentaje_directo)

        def format_signed_pct(x):
            try:
                x_clean = str(x).replace('%','').strip()
                return f"{float(x_clean)*100:+.2f}%"
            except:
                return ""

        economia_dia["% Dow Jones"] = economia_dia["% Dow Jones"].apply(format_signed_pct)
        economia_dia["% S&P500"] = economia_dia["% S&P500"].apply(format_signed_pct)
        economia_dia["% Nasdaq"] = economia_dia["% Nasdaq"].apply(format_signed_pct)


        # Convertir a OrderedDict para frontend
        economia_dict = OrderedDict()
        for col in orden_columnas:
            economia_dict[col] = economia_dia.iloc[0].get(col, "")


    # 📰 Titulares sin repetir medios
    titulares_info = []
    usados_medios = set()

    def agregar_titulares(df_origen, max_count):
        added = 0
        for _, row in df_origen.iterrows():
            medio = row["Fuente"]
            if medio not in usados_medios:
                titulares_info.append({
                    "titulo": row["Título"],
                    "medio": medio,
                    "enlace": row["Enlace"],
                    "idioma": "es"
                })
                usados_medios.add(medio)
                added += 1
            if added >= max_count:
                break

    # 2 nacionales + 2 locales + 2 internacionales + 2 otros = 8 titulares distintos
    agregar_titulares(noticias_nacionales, 2)
    agregar_titulares(noticias_locales, 2)
    agregar_titulares(noticias_internacionales_forzadas, 2)
    agregar_titulares(noticias_otras_forzadas, 2)

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


    return ({
        "resumen": resumen_texto,
        "nube_url": f"/nube/{archivo_nube}",
        "economia": [economia_dict],
        "orden_economia": orden_columnas,
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
@app.route("/pregunta", methods=["POST"])
def pregunta():
    data = request.get_json()
    q = data.get("pregunta", "")
    if not q:
        return jsonify({"respuesta": "No se proporcionó una pregunta."})

    # 1️⃣ Detectar sentimiento deseado
    sentimiento_deseado = detectar_sentimiento_deseado(q)

    # 2️⃣ Detectar rango de fechas o fecha única
    fecha_inicio, fecha_fin = extraer_rango_fechas(q)
    if fecha_inicio and fecha_fin:
        df_filtrado = df[(df["Fecha"].dt.date >= fecha_inicio) & (df["Fecha"].dt.date <= fecha_fin)]
    else:
        fecha_detectada = extraer_fecha(q)
        if fecha_detectada:
            fecha_dt = fecha_detectada
        else:
            fecha_dt = obtener_fecha_mas_reciente(df)
        df_filtrado = df[df["Fecha"].dt.date == fecha_dt]

    # 3️⃣ Extraer entidades
    entidades = extraer_entidades(q)

    # 4️⃣ Aplicar filtros de entidades y sentimiento
    # 4️⃣ Aplicar filtros de entidades y sentimiento sobre todo el rango o fecha
    df_filtrado = filtrar_titulares(df_filtrado, entidades, sentimiento_deseado)


    # 5️⃣ Si no hay resultados
    if df_filtrado.empty:
        if fecha_inicio and fecha_fin:
            return jsonify({"respuesta": f"No encontré noticias relacionadas con tu pregunta entre {fecha_inicio} y {fecha_fin}."})
        else:
            return jsonify({"respuesta": f"No encontré noticias relacionadas con tu pregunta para {fecha_dt}."})

    # 6️⃣ Vectorizar titulares y pregunta
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_filtrado["Título"])
    pregunta_vec = tfidf.transform([q])

    # 7️⃣ Calcular similitudes
    similitudes = cosine_similarity(pregunta_vec, tfidf_matrix).flatten()
    top_indices = similitudes.argsort()[-5:][::-1]  # top 5

    titulares_relevantes = df_filtrado.iloc[top_indices]

    # 8️⃣ Construir prompt
    prompt = "Con base en los siguientes titulares de noticias, responde la pregunta de forma contextual y sin inventar datos, la respuesta debe tener al menos 100 palabras, redactadas en párrafos completos y en tono profesional:\n\n"
    for _, row in titulares_relevantes.iterrows():
        prompt += f"- {row['Título']} ({row['Fuente']})\n"
    prompt += f"\nPregunta: {q}\nRespuesta:"

    # 9️⃣ Llamada a OpenAI con la misma lógica de /resumen
    respuesta = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en análisis de noticias del ramo económico y comercial, con énfasis en aranceles, naves y parques industriales y FIBRAS."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )
    respuesta_gpt = respuesta.choices[0].message.content

    # 🔟 Devolver respuesta y titulares
    titulares_info = [
        {
            "titulo": row["Título"],
            "medio": row["Fuente"],
            "enlace": row["Enlace"]
        }
        for _, row in titulares_relevantes.iterrows()
    ][:5]  # máximo 5

    return jsonify({
        "respuesta": respuesta_gpt,
        "titulares_usados": titulares_info
    })

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
    fecha_dt = pd.to_datetime(fecha_str).date()
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt].copy()

    # Si no hay dato exacto, usar el último disponible para las 4 inflaciones
    for col in ["Inflación Anual MEX", "Inflación Subyacente MEX",
                "Inflación Anual US", "Inflación Subyacente US"]:
        if col in df_economia.columns:
            if economia_dia.empty or economia_dia[col].isnull().all() or economia_dia[col].iloc[0] in ["", None]:
                valor_reciente = df_economia[col].dropna().iloc[-1]
                economia_dia[col] = valor_reciente

    if not economia_dia.empty:
        df_formateada = economia_dia.copy()

        # Columnas en dólares
        for col in ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"]:
            if col in df_formateada.columns:
                df_formateada[col] = pd.to_numeric(df_formateada[col], errors="coerce")
                df_formateada[col] = df_formateada[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

        # Función segura de porcentaje (para TIIEs e Inflaciones)
        def formatear_porcentaje(x):
            if pd.isnull(x):
                return ""
            return f"{(x*100 if abs(x) <= 1 else x):.2f}%"

        # Formateo especial para SOFR
        def format_porcentaje_directo(x):
            try:
                x_clean = str(x).replace('%','').strip()
                return f"{float(x_clean)*100:.2f}%"
            except:
                return ""

        # Formateo especial para índices de Wall Street (con signo)
        def format_signed_pct(x):
            try:
                x_clean = str(x).replace('%','').strip()
                return f"{float(x_clean)*100:+.2f}%"
            except:
                return ""

        # Aplicar formateos
        for col in ["Tasa de Interés Objetivo", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días",
                    "Inflación Anual MEX", "Inflación Subyacente MEX",
                    "Inflación Anual US", "Inflación Subyacente US"]:
            if col in df_formateada.columns:
                df_formateada[col] = pd.to_numeric(df_formateada[col], errors="coerce")
                df_formateada[col] = df_formateada[col].apply(formatear_porcentaje)

        if "SOFR" in df_formateada.columns:
            df_formateada["SOFR"] = df_formateada["SOFR"].apply(format_porcentaje_directo)

        for col in ["% Dow Jones", "% S&P500", "% Nasdaq"]:
            if col in df_formateada.columns:
                df_formateada[col] = df_formateada[col].apply(format_signed_pct)

        # Pasar a OrderedDict
        economia_dict = OrderedDict()
        for col in df_formateada.columns[1:]:
            economia_dict[col] = df_formateada.iloc[0][col]

        # 🔹 Construcción manual en filas
        filas = [
            ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"],
            ["Tasa de Interés Objetivo", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días"],
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
    remitente = "ldsantiagovidargas.93@gmail.com"
    password = os.environ.get("GMAIL_PASSWORD_APP")
    destinatario = email

    msg = MIMEMultipart()
    msg["From"] = remitente
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
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(remitente, password)
        server.sendmail(remitente, destinatario, msg.as_string())
        server.quit()
        return jsonify({"mensaje": f"✅ Correo enviado a {destinatario}"})
    except Exception as e:
        return jsonify({"mensaje": f"❌ Error al enviar correo: {e}"})

@app.route("/nube/<filename>")
def serve_nube(filename):
    return send_from_directory("nubes", filename)

@app.route("/fechas")
def get_fechas():
    # Obtener todas las fechas únicas, ordenadas de más reciente a más antigua
    fechas_unicas = sorted(df["Fecha"].dt.date.unique(), reverse=True)
    fechas_str = [f.strftime("%Y-%m-%d") for f in fechas_unicas]
    return jsonify(fechas_str)

# ------------------------------
# 📑 Endpoint para análisis semanal
# ------------------------------
@app.route("/reporte_semanal", methods=["GET"])
def reporte_semanal():
    carpeta = "reporte_semanal"
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
