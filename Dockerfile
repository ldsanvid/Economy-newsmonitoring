# --- Etapa base ---
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/Mexico_City

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto
COPY . .

# ✅ Copiar explícitamente los recursos de datos
COPY faiss_index/ /app/faiss_index/
COPY reporte_semanal/ /app/reporte_semanal/
COPY noticias_fondo_fuentes_rango_03-07-2025.csv /app/
COPY tipo_cambio_tasas_interes.xlsx /app/

>>>>>>> dockerfix

# Forzar rebuild completo
RUN echo "forcing rebuild $(date)"

# (Opcional) verificar en logs que sí existan los archivos dentro del contenedor
RUN echo "✅ Archivos dentro de /app:" && ls -R /app

# Exponer el puerto del backend
EXPOSE 5000

# Comando de arranque
CMD ["gunicorn", "backend_final:app", "--bind", "0.0.0.0:5000"]

