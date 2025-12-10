FROM python:3.11-slim

WORKDIR /app

# Устанавливаем переменные окружения для оптимизации
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers
ENV PIP_NO_CACHE_DIR=1

# Install dependencies (используем production requirements)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application
COPY main.py .

# Expose port
EXPOSE 8082

# Run with uvicorn (ограничиваем workers для экономии памяти)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1"]

