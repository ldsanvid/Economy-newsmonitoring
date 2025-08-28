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
df_infl_mx = pd.read_excel("tipo de cambio y tasas de interés.xlsx", sheet_name="InflaciónMEX")

# Unificar fechas
for df_tmp in [df_sofr, df_wall, df_infl_us, df_infl_mx]:
    df_tmp["Fecha"] = pd.to_datetime(df_tmp["Fecha"], errors="coerce").dt.date

# Unir con df_economia
df_economia["Fecha"] = pd.to_datetime(df_economia["Fecha"], errors="coerce").dt.date
df_economia = df_economia.merge(df_sofr[["Fecha", "SOFR"]], on="Fecha", how="left")
df_economia = df_economia.merge(df_infl_us[["Fecha", "Inflación USA"]], on="Fecha", how="left")
df_economia = df_economia.merge(df_infl_mx[["Fecha", "Inflación México"]], on="Fecha", how="left")
df_economia = df_economia.merge(df_wall[["Fecha", "% Dow Jones", "% S&P500", "% Nasdaq"]], on="Fecha", how="left")

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
- Javier Millei es el actual Presidente de Argentina.
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
        "Millei": ["presidente de argentina"],
        "Von Der Leyen": ["presidenta de la comisión europea"],
        "Petro": ["presidente de colombia"]
    }
    lugares_dict = {
        "Nuevo León": ["nl", "monterrey"],
        "Ciudad de México": ["cdmx", "capital mexicana"],
        "Reino Unido": ["gran bretaña", "inglaterra"],
        "Estados Unidos": ["eeuu", "eua", "usa", "eu"]
    }
    categorias_dict = {
        "Aranceles": ["arancel","tarifas", "restricciones comerciales"],
        "Parque Industrial": ["zona industrial","parque industrial"],
        "Fibra": ["fideicomiso inmobiliario", "fibras"],
        "Fusiones": ["adquisiciones", "compras empresariales"],
        "Naves Industriales": ["inmuebles industriales","nave industrial","bodegas industriales"],
        "Real Estate": ["mercado inmobiliario"],
        "Construcción Industrial": ["obra industrial"],
        "Sector Industrial": ["industria pesada", "manufactura"]
    }
    encontrados = {"personajes": [], "lugares": [], "categorias": []}

    for nombre, sinonimos in personajes_dict.items():
        if any(s in texto_lower for s in [nombre.lower()] + sinonimos):
            encontrados["personajes"].append(nombre)

    for lugar, sinonimos in lugares_dict.items():
        if any(s in texto_lower for s in [lugar.lower()] + sinonimos):
            encontrados["lugares"].append(lugar)

    for cat, sinonimos in categorias_dict.items():
        if any(s in texto_lower for s in [cat.lower()] + sinonimos):
            encontrados["categorias"].append(cat)

    return encontrados

# 5️⃣ Filtrar titulares por fecha, entidades y sentimiento
def filtrar_titulares(fecha, entidades, sentimiento_deseado):
    noticias_fecha = df[df["Fecha"].dt.date == fecha]
    if noticias_fecha.empty:
        return pd.DataFrame()  # 🔹 Devuelve DataFrame vacío si no hay noticias

    filtro = noticias_fecha.copy()

    if entidades["personajes"]:
        filtro = filtro[filtro["Título"].str.lower().apply(lambda t: any(p.lower() in t for p in entidades["personajes"]))]
    if entidades["lugares"]:
        filtro = filtro[filtro["Cobertura"].str.lower().apply(lambda c: any(l.lower() in c for l in entidades["lugares"]))]
    if entidades["categorias"]:
        filtro = filtro[filtro["Término"].str.lower().apply(lambda cat: any(e.lower() in cat for e in entidades["categorias"]))]
    if sentimiento_deseado:
        filtro = filtro[filtro["Sentimiento"] == sentimiento_deseado]

    return filtro  # 🔹 Devuelve DataFrame, no lista

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
        "ayer", "prácticamente", "sustancialmente", "busca", "cómo", "qué", "días"
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

#/resumen!!!!!!!
@app.route("/resumen", methods=["POST"])
def resumen():
    data = request.get_json()
    fecha_str = data.get("fecha")
    if not fecha_str:
        return jsonify({"error": "Debe especificar una fecha"}), 400

    # Filtrar noticias de la fecha
    fecha_dt = pd.to_datetime(fecha_str, errors="coerce").date()
    noticias_dia = df[df["Fecha"].dt.date == fecha_dt]
    if noticias_dia.empty:
        return jsonify({"error": f"No hay noticias para la fecha {fecha_str}"}), 404

    # Clasificación por cobertura
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

Cuarto párrafo: Por último, resume de forma general las noticias que no están relacionadas con aranceles y que tienen que ver con fibras, naves industriales, parques industriales, hub logístico, hub industrial, real estate industrial, sector industrial o sector mobiliario industrial. Evita repetir noticias mencionadas en los otros párrafos ni inventes cosas. Recuerda, temas no arancelarios. Empieza diciendo "finalmente en otros temas económicos", sin recalcar de que se trata de noticias del ámbito local o nacional..

Noticias nacionales:
{contexto_nacional}

Noticias locales:
{contexto_local}

Noticias internacionales:
{contexto_internacional}

Noticias no relacionadas con aranceles:
{contexto_otros_temas}
    """

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

# ☁️ Generar nube de palabras
    os.makedirs("nubes", exist_ok=True)  # crea carpeta si no existe
    archivo_nube = f"nube_{fecha_str}.png"
    archivo_nube_path = os.path.join("nubes", archivo_nube)
    generar_nube(noticias_dia["Título"].tolist(), archivo_nube_path)
    # 💾 Guardar resumen en carpeta "resumenes"
    os.makedirs("resumenes", exist_ok=True)
    archivo_resumen = f"resumen_{fecha_str}.txt"
    archivo_resumen_path = os.path.join("resumenes", archivo_resumen)
    with open(archivo_resumen_path, "w", encoding="utf-8") as f:
        f.write(resumen_texto)


    # 📊 Indicadores económicos
    # Filtrar datos económicos
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt]
    # Si la inflación USA está vacía en el día seleccionado, usar el valor más reciente disponible
    if economia_dia["Inflación USA"].isnull().all() or economia_dia["Inflación USA"].iloc[0] in ["", None]:
        inflacion_usa_reciente = df_economia["Inflación USA"].dropna().replace("", np.nan).dropna().iloc[-1]
        economia_dia["Inflación USA"] = inflacion_usa_reciente


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
            "Inflación USA",
            "Inflación México",
            "% Dow Jones",
            "% S&P500",
            "% Nasdaq"
        ]
        def format_porcentaje_directo(x):
            try:
                x_clean = str(x).replace('%','').strip()
                return f"{float(x_clean)*100:.2f}%"
            except:
                return ""
        # Formato para nuevos indicadores
        economia_dia["SOFR"] = economia_dia["SOFR"].apply(format_porcentaje_directo)
        economia_dia["Inflación USA"] = economia_dia["Inflación USA"].apply(format_porcentaje_directo)
        economia_dia["Inflación México"] = economia_dia["Inflación México"].apply(format_porcentaje_directo)


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
                    "enlace": row["Enlace"]
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
                    "enlace": row["Enlace"]
                })
                usados_medios_en.add(medio)
            if len(titulares_info_en) >= 8:
                break


    return jsonify({
        "resumen": resumen_texto,
        "nube_url": f"/nube/{archivo_nube}",
        "economia": [economia_dict],
        "orden_economia": orden_columnas,
        "titulares": titulares_info,
        "titulares_en": titulares_info_en  # 👈 nuevo bloque de titulares en inglés
    })


#pregunta!!!!    
@app.route("/pregunta", methods=["POST"])
def pregunta():
    data = request.get_json()
    q = data.get("pregunta", "")
    if not q:
        return jsonify({"respuesta": "No se proporcionó una pregunta."})

    # 1️⃣ Detectar sentimiento deseado
    sentimiento_deseado = detectar_sentimiento_deseado(q)

    # 2️⃣ Extraer fecha de la pregunta
    fecha_detectada = extraer_fecha(q)
    if fecha_detectada:
        fecha_dt = fecha_detectada  # ya es un objeto date
    else:
        fecha_dt = obtener_fecha_mas_reciente(df)  # usamos la más reciente

    # 3️⃣ Extraer entidades
    entidades = extraer_entidades(q)

    # 4️⃣ Filtrar titulares
    df_filtrado = filtrar_titulares(fecha_dt, entidades, sentimiento_deseado)

    # 5️⃣ Si no hay resultados
    if df_filtrado.empty:
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
    # 🔟 Devolver respuesta y titulares usados (entre 1 y 5)
    titulares_info = [
        {
            "titulo": row["Título"],
            "medio": row["Fuente"],
            "enlace": row["Enlace"]
        }
        for _, row in titulares_relevantes.iterrows()
    ]

    # Garantizar que haya mínimo 1 y máximo 5 titulares
    if len(titulares_info) == 0:
        # Si por alguna razón no hubiera titulares relevantes
        titulares_info = [
            {
                "titulo": row["Título"],
                "medio": row["Fuente"],
                "enlace": row["Enlace"]
            }
            for _, row in df_filtrado.head(1).iterrows()
        ]
    else:
        titulares_info = titulares_info[:5]

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

    if not email or not fecha_str:
        return jsonify({"mensaje": "Debes proporcionar correo y fecha"}), 400

    # 📝 Resumen guardado
    archivo_resumen = os.path.join("resumenes", f"resumen_{fecha_str}.txt")
    if not os.path.exists(archivo_resumen):
        return jsonify({"mensaje": f"No hay resumen disponible para {fecha_str}"}), 404

    with open(archivo_resumen, "r", encoding="utf-8") as f:
        resumen_texto = f.read()

    # ☁️ Nube
    archivo_nube = os.path.join("nubes", f"nube_{fecha_str}.png")

# 📰 Titulares por idioma (unificados en una sola variable HTML)
    titulares_html = ""  # Inicializamos la variable

    titulares_es = df[(df["Fecha"].dt.date == fecha_dt) & (df["Idioma"].str.lower() == "es")]
    titulares_en = df[(df["Fecha"].dt.date == fecha_dt) & (df["Idioma"].str.lower().isin(["en", "ingles", "inglés"]))]

    # Titulares en español
    if not titulares_es.empty:
        titulares_html += "<h3>📰 Principales titulares en español</h3><ul>"
        for _, row in titulares_es.head(8).iterrows():
            titulo = row["Título"]
            enlace = row["Enlace"]
            medio = row["Fuente"]
            titulares_html += f'<li><a href="{enlace}" target="_blank">{titulo}</a> — <em>{medio}</em></li>'
        titulares_html += "</ul>"

    # Titulares en inglés
    if not titulares_en.empty:
        titulares_html += "<h3>🌎 Principales titulares en inglés</h3><ul>"
        for _, row in titulares_en.head(8).iterrows():
            titulo = row["Título"]
            enlace = row["Enlace"]
            medio = row["Fuente"]
            titulares_html += f'<li><a href="{enlace}" target="_blank">{titulo}</a> — <em>{medio}</em></li>'
        titulares_html += "</ul>"

    # Si no hay titulares de ningún tipo
    if not titulares_html:
        titulares_html = "<p>No se encontraron titulares relevantes para esta fecha.</p>"


    fecha_dt = pd.to_datetime(fecha_str).date()
    economia_dia = df_economia[df_economia["Fecha"] == fecha_dt].copy()
    # Si la inflación USA está vacía ese día, usar la más reciente disponible
    if "Inflación USA" in economia_dia.columns and economia_dia["Inflación USA"].isnull().all():
        inflacion_usa_reciente = df_economia["Inflación USA"].dropna().iloc[-1]
        economia_dia["Inflación USA"] = inflacion_usa_reciente
    
    if not economia_dia.empty:
        df_formateada = economia_dia.copy()

            # Columnas en dólares
        cols_dolar = ["Tipo de Cambio FIX", "Nivel máximo", "Nivel mínimo"]
        for col in cols_dolar:
            if col in df_formateada.columns:
                df_formateada[col] = df_formateada[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

            # Columnas en porcentaje
        cols_porcentaje = [
                "Tasa de Interés Objetivo", "TIIE 28 días", "TIIE 91 días", "TIIE 182 días",
                "SOFR", "Inflación USA", "Inflación México",
                "% Dow Jones", "% S&P500", "% Nasdaq"
            ]
        for col in cols_porcentaje:
            if col in df_formateada.columns:
                df_formateada[col] = df_formateada[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")

                tabla_html = df_formateada.to_html(index=False, border=1)
            else:
                tabla_html = "<p>No hay datos económicos</p>"


    # ---- CONFIGURACIÓN DEL CORREO ----
    remitente = "ldsantiagovidargas.93@gmail.com"
    password = os.environ.get("GMAIL_PASSWORD_APP")  # contraseña de aplicación guardada en Render
    destinatario = email

    msg = MIMEMultipart()
    msg["From"] = remitente
    msg["To"] = destinatario
    msg["Subject"] = f"Resumen de noticias {fecha_str}"

    cuerpo = f"""
    <h2>Resumen de noticias del {fecha_str}</h2>
    <p style="text-align: justify;">{resumen_texto}</p>
    <h3>📊 Indicadores económicos</h3>
    {tabla_html}
    <h3>Principales titulares en español</h3>
    {titulares_es}

    <h3>Principales titulares en inglés</h3>
    {titulares_en}

    <p>Palabras más repetidas en los titulares:</p>
    <img src="cid:nube" alt="Nube de palabras" style="width:100%; max-width:600px; margin-top:20px;" />
    <p>Adjunto encontrarás la nube de palabras en formato de imagen.</p>
    """

    msg.attach(MIMEText(cuerpo, "html"))
    # 📎 Incrustar imagen de nube de palabras en el cuerpo del correo
    if os.path.exists(archivo_nube):
        with open(archivo_nube, "rb") as img_file:
            imagen = MIMEImage(img_file.read())
            imagen.add_header("Content-ID", "<nube>")
            imagen.add_header("Content-Disposition", "inline", filename=archivo_nube)
            msg.attach(imagen)


    # Adjuntar nube
    imagen_html = ""
    if os.path.exists(archivo_nube):
        with open(archivo_nube, "rb") as f:
            imagen_bytes = f.read()
            imagen_b64 = base64.b64encode(imagen_bytes).decode("utf-8")
            imagen_html = f'<p><img src="data:image/png;base64,{imagen_b64}" alt="Nube de palabras" width="600"/></p>'

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

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
