FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app/ ./app/
COPY faiss_index/ ./faiss_index/
COPY image_index/ ./image_index/
COPY knowledge_base/ ./knowledge_base/
COPY step2_rag_engine.py .
COPY backend/static/ ./static/

# Create symlink for relative imports to work
RUN mkdir -p /app && touch /app/__init__.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
