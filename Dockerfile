# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# Dockerfile for HuggingFace Spaces
# Version: 2.1.0 (Production-Ready)

FROM python:3.11-slim

# Create user (required by HuggingFace)
RUN useradd -m -u 1000 user
USER user

# Set PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# ============== Environment Variables ==============

# Base settings
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/user/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence_transformers

# Model settings
ENV EMBEDDING_MODEL=ai-forever/ru-en-RoSBERTa
ENV EMBEDDING_DIMENSIONS=768

# Limits (production-ready)
ENV MAX_BATCH_SIZE=128
ENV MAX_TEXT_LENGTH=10000
ENV MAX_CONCURRENT_REQUESTS=6
ENV ENCODE_TIMEOUT_SECONDS=30.0

# Rate limiting
ENV RATE_LIMIT=100/minute
ENV RATE_LIMIT_BATCH=20/minute

# Cache settings
ENV CACHE_ENABLED=true
ENV CACHE_TTL_SECONDS=3600
ENV CACHE_MAX_SIZE=10000

# Security (переопределите в production!)
ENV ALLOWED_ORIGINS=*

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user main.py .

# Expose port 7860 (HuggingFace Spaces standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

