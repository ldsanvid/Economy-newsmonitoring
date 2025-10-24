import pandas as pd
import numpy as np
import faiss
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def update_embeddings(csv_path="noticias_fondo con todas las fuentes_rango_03-07-2025.csv"):
    os.makedirs("faiss_index", exist_ok=True)

    df_new = pd.read_csv(csv_path, encoding="utf-8")
    meta_path = "faiss_index/noticias_metadata.csv"
    emb_path = "faiss_index/noticias_embeddings.npy"
    index_path = "faiss_index/noticias_index.faiss"

    # Si ya existe índice, lo cargamos
    if os.path.exists(meta_path) and os.path.exists(emb_path):
        df_old = pd.read_csv(meta_path)
        existing_titles = set(df_old["Título"])
        df_to_embed = df_new[~df_new["Título"].isin(existing_titles)]
        print(f"📈 Nuevos titulares detectados: {len(df_to_embed)}")
    else:
        df_to_embed = df_new
        print(f"🧠 Generando embeddings iniciales: {len(df_to_embed)}")

    if df_to_embed.empty:
        print("✅ No hay nuevos titulares.")
        return

    # Crear embeddings nuevos
    batch_size = 1000
    embeddings = []
    for i in range(0, len(df_to_embed), batch_size):
        batch = df_to_embed["Título"].iloc[i:i+batch_size].tolist()
        resp = client.embeddings.create(model="text-embedding-3-large", input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    new_embeddings = np.array(embeddings).astype("float32")

    # Cargar o crear índice
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        old_emb = np.load(emb_path)
        updated_emb = np.vstack([old_emb, new_embeddings])
        index.add(new_embeddings)
    else:
        updated_emb = new_embeddings
        index = faiss.IndexFlatL2(new_embeddings.shape[1])
        index.add(new_embeddings)

    # Guardar actualizados
    faiss.write_index(index, index_path)
    np.save(emb_path, updated_emb)

    # Unir metadatos
    df_final = pd.concat([df_new], ignore_index=True)
    df_final.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("✅ Embeddings actualizados y guardados.")

if __name__ == "__main__":
    update_embeddings()
