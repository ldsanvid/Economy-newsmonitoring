import pandas as pd
import numpy as np
import faiss
from datetime import datetime, timedelta
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

RES_DIR = "faiss_index"
META_PATH = os.path.join(RES_DIR, "resumenes_metadata.csv")
INDEX_PATH = os.path.join(RES_DIR, "resumenes_index.faiss")

os.makedirs(RES_DIR, exist_ok=True)

def cargar_indice_resumenes():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(3072)  # tamaño aprox del embedding de text-embedding-3sí-large
    return index

def obtener_resumen_mas_reciente(fecha_actual):
    if not os.path.exists(META_PATH):
        return ""
    df = pd.read_csv(META_PATH)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    anteriores = df[df["Fecha"] < pd.Timestamp(fecha_actual)]
    if anteriores.empty:
        return ""
    mas_reciente = anteriores.sort_values("Fecha", ascending=False).iloc[0]
    return mas_reciente["Resumen"]

def guardar_resumen_vectorizado(fecha, texto):
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=texto
    ).data[0].embedding

    df_row = pd.DataFrame([{"Fecha": fecha, "Resumen": texto}])
    if os.path.exists(META_PATH):
        df = pd.read_csv(META_PATH)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(META_PATH, index=False, encoding="utf-8-sig")

    index = cargar_indice_resumenes()
    index.add(np.array([emb]).astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ Resumen vectorizado y guardado ({fecha})")

def buscar_resumen_mas_relacionado(texto):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return ""
    index = cargar_indice_resumenes()
    df = pd.read_csv(META_PATH)
    emb = client.embeddings.create(model="text-embedding-3-large", input=texto).data[0].embedding
    _, idx = index.search(np.array([emb]).astype("float32"), k=1)
    return df.iloc[idx[0][0]]["Resumen"]
