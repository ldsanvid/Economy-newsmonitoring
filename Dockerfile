# --- Etapa base ---
FROM python:3.10-slim

# Evitar buffering y usar UTF-8
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/Mexico_City

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto
COPY . .

# Exponer el puerto
EXPOSE 5000

# Comando de arranque para Render
CMD ["gunicorn", "backend_final:app", "--bind", "0.0.0.0:5000"]
