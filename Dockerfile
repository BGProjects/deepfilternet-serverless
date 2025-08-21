# DeepFilterNet3 Serverless - RTX 5090 Optimized for RunPod
# Multi-stage build with embedded models for optimal performance
# Following RunPod best practices for serverless workers

# ========================================
# Stage 1: Base CUDA Development Environment  
# ========================================
# Using devel image for better CUDA compatibility across versions
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# RunPod optimization environment variables
ENV RUNPOD_INIT_TIMEOUT=800
ENV USE_FLASH_ATTENTION=true
ENV RUNPOD_WORKER_TIMEOUT=300

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libsndfile1 \
    libsndfile1-dev \
    libfftw3-dev \
    libasound2-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ========================================
# Stage 2: Dependencies Installation
# ========================================
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch nightly with RTX 5090 (sm_120) support first
RUN pip install --no-cache-dir --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Python dependencies with optimizations  
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install ONNX Runtime GPU with CUDA 12.4 support (updated version)
RUN pip install --no-cache-dir onnxruntime-gpu==1.18.1 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Preload CUDA libraries for ONNX Runtime compatibility
RUN python -c "import torch; import onnxruntime; onnxruntime.preload_dlls() if hasattr(onnxruntime, 'preload_dlls') else None"

# Verify ONNX Runtime installation
RUN python -c "import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__); print('Available providers:', ort.get_available_providers())"

# ========================================
# Stage 3: Application Setup
# ========================================
FROM dependencies AS application

# Copy application source code
COPY src/ /app/src/
COPY models/ /app/models/

# Create necessary directories
RUN mkdir -p /app/logs /app/temp /tmp/trt_cache

# Set permissions
RUN chmod -R 755 /app/src/ && \
    chmod -R 644 /app/models/

# ========================================  
# Stage 4: Production Image
# ========================================
FROM application AS production

# Set working directory
WORKDIR /app

# RunPod + RTX 5090 Optimizations
# Multiple CUDA version compatibility (RunPod best practice)
ENV CUDA_VERSION=12.4
ENV CUDA_COMPAT_VERSIONS="12.1,12.2,12.3,12.4,12.5,12.6"
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_MAXSIZE=4294967296
ENV CUDA_DEVICE_MAX_CONNECTIONS=32

# RunPod Serverless Optimization Variables
ENV RUNPOD_INIT_TIMEOUT=800
ENV RUNPOD_WORKER_REFRESH=true
ENV RUNPOD_ERROR_HANDLER=comprehensive

# Flash Attention for RTX 5090 (RunPod recommended)
ENV USE_FLASH_ATTENTION=true
ENV FLASH_ATTENTION_VERSION=2

# ONNX Runtime Optimizations for RTX 5090
ENV OMP_NUM_THREADS=12
ENV ORT_TENSORRT_MAX_WORKSPACE_SIZE=17179869184
ENV ORT_TENSORRT_FP16_ENABLE=1
ENV ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
ENV ORT_TENSORRT_ENGINE_CACHE_PATH=/tmp/trt_cache

# Memory optimizations - RTX 5090 32GB VRAM (RunPod optimized)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,expandable_segments:True,roundup_power2_divisions:16
ENV CUDA_MEMORY_FRACTION=0.85
ENV GPU_MEMORY_UTILIZATION=0.85
ENV PYTORCH_ALLOC_CONF=max_split_size_mb:2048

# RunPod FlashBoot optimization
ENV RUNPOD_FLASHBOOT=true
ENV RUNPOD_KEEP_WARM=300

# Logging configuration
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app/src

# RunPod compatible health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=5 \
    CMD python -c "import sys; sys.path.append('/app/src'); from utils import log_system_info; log_system_info(); print('RunPod Worker Health: OK')" || exit 1

# Set the default command
CMD ["python", "/app/src/handler.py"]

# ========================================
# Build Information
# ========================================
LABEL maintainer="BGProjects"
LABEL description="DeepFilterNet3 Serverless Audio Enhancement - RTX 5090 Optimized with CUDA 12.8"
LABEL version="2.0.0"
LABEL cuda_version="12.4"
LABEL onnxruntime_version="1.18.1"
LABEL pytorch_version="2.8.0-nightly"
LABEL python_version="3.10"

# RunPod Serverless Labels (RunPod best practice)
LABEL runpod.serverless=true
LABEL runpod.gpu="RTX5090"
LABEL runpod.gpu_compatibility="RTX4090,RTX5080,RTX5090,H100,A100"
LABEL runpod.memory_requirement="8GB"
LABEL runpod.processing_type="audio_enhancement"
LABEL runpod.cuda_capability="sm_80,sm_86,sm_89,sm_90,sm_120"
LABEL runpod.cuda_versions="12.1,12.2,12.3,12.4,12.5,12.6"
LABEL runpod.worker_type="optimized"
LABEL runpod.model_embedded=true
LABEL runpod.flashboot_compatible=true