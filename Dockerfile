# FramePack Dockerfile for RunPod Serverless
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install PyTorch with CUDA 12.6
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    Pillow

# Copy FramePack application files
COPY . .

# Set environment variables for model caching to network volume
# CRITICAL: Set TMPDIR to network volume so HuggingFace downloads don't fill local disk
ENV HF_HOME=/runpod-volume/huggingface
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface/transformers
ENV HF_DATASETS_CACHE=/runpod-volume/huggingface/datasets
ENV TORCH_HOME=/runpod-volume/torch
ENV TMPDIR=/runpod-volume/tmp
ENV TEMP=/runpod-volume/tmp
ENV TMP=/runpod-volume/tmp
ENV PYTHONUNBUFFERED=1

# Run the handler
CMD ["python", "/app/handler.py"]
