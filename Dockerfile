# FramePack Dockerfile for RunPod Serverless
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install PyTorch with CUDA 12.6
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod serverless SDK
RUN pip install runpod

# Copy application files
COPY . .

# Create handler for RunPod
COPY handler.py /app/handler.py

# Run the handler
CMD ["python", "/app/handler.py"]
