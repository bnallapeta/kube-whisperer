# Build stage for dependencies
FROM --platform=$BUILDPLATFORM python:3.11-slim as deps

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    cmake \
    pkg-config \
    ffmpeg \
    libmagic1 \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip --version \
    && gcc --version

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in layers for better caching
COPY requirements.txt /tmp/requirements.txt

# Install CPU-only PyTorch first (faster to build)
RUN pip3 install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.109.2 \
    uvicorn[standard]==0.27.1 \
    python-multipart==0.0.9 \
    pydantic==2.6.1 \
    pydantic-core==2.16.2 \
    python-magic==0.4.27

# Install remaining dependencies
RUN pip3 install --no-cache-dir \
    openai-whisper==20231117 \
    numpy==1.24.3 \
    tqdm==4.66.1 \
    typing-extensions>=4.8.0 \
    prometheus-client==0.19.0 \
    structlog==24.1.0 \
    psutil==5.9.8

# Runtime stage
FROM --platform=$TARGETPLATFORM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    gcc \
    git \
    cmake \
    pkg-config \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM --platform=$TARGETPLATFORM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gcc \
    ffmpeg \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create app structure and copy files properly
COPY src/ /app/src/
COPY samples/ /app/samples/

# Move serve.py to the correct location
COPY src/serve.py /app/serve.py

# Add /app to the Python path
ENV PYTHONPATH=/app

# Create directories for temporary and cache files
RUN mkdir -p /tmp/whisper_audio && \
    chmod 777 /tmp/whisper_audio && \
    mkdir -p /tmp/whisper_cache && \
    chmod 777 /tmp/whisper_cache && \
    mkdir -p /tmp/whisper_cache/models && \
    chmod 777 /tmp/whisper_cache/models && \
    mkdir -p /tmp/whisper/temp && \
    chmod 777 /tmp/whisper/temp

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONFAULTHANDLER=1 \
    PYTHON_WARNINGS=on \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    XDG_CACHE_HOME=/tmp/whisper_cache \
    MODEL_DOWNLOAD_ROOT=/tmp/whisper_cache/models

EXPOSE 8000

CMD ["python3", "-X", "faulthandler", "serve.py"]