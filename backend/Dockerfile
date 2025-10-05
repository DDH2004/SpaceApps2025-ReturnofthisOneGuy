# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi>=0.104.0 uvicorn[standard]>=0.24.0 websockets>=12.0
RUN pip install --no-cache-dir redis>=5.0.0 celery>=5.3.0 pydantic>=2.5.0
RUN pip install --no-cache-dir python-dotenv>=1.0.0 python-multipart>=0.0.6
RUN pip install --no-cache-dir numpy pandas scikit-learn joblib

# Copy application code
COPY . .

# Create directory for ML models
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
