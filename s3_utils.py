import os
import boto3
from botocore.client import Config

# 🌐 Variables de entorno que configuraremos en Render
BUCKET = os.environ.get("S3_BUCKET")
REGION = os.environ.get("S3_REGION", "us-east-2")
ACCESS_KEY = os.environ.get("S3_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")

# ⚙️ Cliente S3
session = boto3.session.Session()
s3 = session.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4")
)

# 📁 Archivos esperados por tu backend
FAISS_DIR = "faiss_index"
FILES = [
    "noticias_index.faiss",
    "noticias_metadata.csv",
    "resumenes_index.faiss",
    "resumenes_metadata.csv",
]

def s3_download_all(prefix="faiss_index"):
    """Descarga los archivos FAISS/CSV desde S3 al servidor local"""
    os.makedirs(FAISS_DIR, exist_ok=True)
    for fname in FILES:
        key = f"{prefix}/{fname}"
        local = os.path.join(FAISS_DIR, fname)
        try:
            s3.download_file(BUCKET, key, local)
            print(f"✅ Descargado desde S3: {key}")
        except Exception as e:
            print(f"⚠️  No se pudo descargar {key}: {e}")

def s3_upload(fname, prefix="faiss_index"):
    """Sube un archivo individual al bucket S3"""
    path = os.path.join(FAISS_DIR, fname)
    key = f"{prefix}/{fname}"
    try:
        s3.upload_file(path, BUCKET, key)
        print(f"☁️ Subido a S3: {key}")
    except Exception as e:
        print(f"❌ Error subiendo {path}: {e}")
