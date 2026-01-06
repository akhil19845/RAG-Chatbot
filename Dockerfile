# Dockerfile (Python 3.11 - recommended for stability)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies (Linux equivalents of brew installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libmagic1 \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    qpdf \
    libheif-dev \
    libjpeg-dev \
    libpng-dev \
    pkg-config \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from your working environment
COPY requirements_full.txt /app/requirements_full.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements_full.txt

# Copy full application source code
COPY . /app

# Expose backend port
EXPOSE 8000

# Default command for API (overridden in docker-compose for dev reload)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]