FROM python:3.11-slim

# Install system dependencies in one layer and clean up immediately
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download OCR and ML models during build to avoid runtime downloads
COPY preload_models.py .
RUN python preload_models.py || echo "Model pre-loading completed with warnings"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/vector_store /app/faiss_index

# Expose port
EXPOSE 8000

# Run the application in development mode
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]