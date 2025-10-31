import os
import boto3
import botocore.client


# 🌐 Variables de entorno (las que ya pusiste en Render)
BUCKET = os.environ.get("R2_BUCKET")
ENDPOINT = os.environ.get("R2_ENDPOINT")
ACCESS_KEY = os.environ.get("R2_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

# ⚙️ Cliente S3 compatible con Cloudflare R2
session = boto3.session.Session()
config = botocore.client.Config(signature_version="s3v4")

s3 = session.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=config,
    verify=False
)


FAISS_DIR = "faiss_index"
FILES = [
    "noticias_index.faiss",
    "noticias_metadata.csv",
    "resumenes_index.faiss",
    "resumenes_metadata.csv",
]

def r2_download_all(prefix="faiss_index"):
    """Descarga los archivos clave FAISS/CSV desde R2 al servidor local"""
    os.makedirs(FAISS_DIR, exist_ok=True)
    for fname in FILES:
        key = f"{prefix}/{fname}"
        local = os.path.join(FAISS_DIR, fname)
        try:
            s3.download_file(BUCKET, key, local)
            print(f"☁️  Descargado desde R2: {key}")
        except Exception as e:
            print(f"⚠️  No se pudo descargar {key}: {e}")

def r2_upload(fname, prefix="faiss_index"):
    """Sube un archivo individual al bucket R2"""
    path = os.path.join(FAISS_DIR, fname)
    key = f"{prefix}/{fname}"
    try:
        s3.upload_file(path, BUCKET, key)
        print(f"☁️  Subido a R2: {key}")
    except Exception as e:
        print(f"❌ Error subiendo {path}: {e}")
