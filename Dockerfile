# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# Dockerfile for HuggingFace Spaces

FROM python:3.11-slim

# Create user (required by HuggingFace)
RUN useradd -m -u 1000 user
USER user

# Set PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/home/user/.cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence_transformers
ENV HF_HOME=/home/user/.cache/huggingface

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user main.py .

# Expose port 7860 (HuggingFace Spaces standard)
EXPOSE 7860

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

