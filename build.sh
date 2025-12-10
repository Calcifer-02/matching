#!/bin/bash
echo "Build completed successfully!"

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
echo "Pre-downloading embedding model to reduce cold start time..."

pip install --no-cache-dir -r requirements.txt
cd embedding-service
echo "Installing Python dependencies..."

# Render build script

