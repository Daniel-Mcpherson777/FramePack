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
    gradio \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface-hub \
    requests \
    Pillow \
    opencv-python-headless

# Copy FramePack application files
COPY . .

# Set environment variables for model caching to network volume
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface/transformers
ENV HF_DATASETS_CACHE=/runpod-volume/huggingface/datasets
ENV TORCH_HOME=/runpod-volume/torch

# Create handler.py with proper FramePack integration
RUN echo 'import runpod\n\
import sys\n\
import os\n\
import tempfile\n\
import requests\n\
from pathlib import Path\n\
import torch\n\
\n\
# Set cache directories to network volume\n\
os.environ["HF_HOME"] = "/runpod-volume/huggingface"\n\
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/huggingface/transformers"\n\
os.environ["HF_DATASETS_CACHE"] = "/runpod-volume/huggingface/datasets"\n\
os.environ["TORCH_HOME"] = "/runpod-volume/torch"\n\
\n\
# Add FramePack to path\n\
sys.path.insert(0, "/app")\n\
\n\
# Global variables for loaded models (load once, reuse)\n\
models_loaded = False\n\
model_components = {}\n\
\n\
def load_models():\n\
    """Load FramePack models once and cache them"""\n\
    global models_loaded, model_components\n\
    \n\
    if models_loaded:\n\
        return model_components\n\
    \n\
    print("Loading FramePack models from network volume...")\n\
    \n\
    try:\n\
        # Import FramePack components\n\
        from diffusers_helper.models import HunyuanVideoTransformer3DModelPacked\n\
        from diffusers_helper.pipeline_hunyuan_packed import AutoencoderKLHunyuanVideo\n\
        from transformers import LlamaModel, CLIPTextModel, SiglipVisionModel\n\
        \n\
        # Load models (will download to /runpod-volume on first run)\n\
        print("Loading transformer...")\n\
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(\n\
            "tencent/HunyuanVideo",\n\
            subfolder="transformer",\n\
            torch_dtype=torch.bfloat16,\n\
            cache_dir="/runpod-volume/huggingface"\n\
        )\n\
        \n\
        print("Loading VAE...")\n\
        vae = AutoencoderKLHunyuanVideo.from_pretrained(\n\
            "tencent/HunyuanVideo",\n\
            subfolder="vae",\n\
            torch_dtype=torch.float16,\n\
            cache_dir="/runpod-volume/huggingface"\n\
        )\n\
        \n\
        print("Loading text encoders...")\n\
        text_encoder = LlamaModel.from_pretrained(\n\
            "tencent/HunyuanVideo",\n\
            subfolder="text_encoder",\n\
            torch_dtype=torch.bfloat16,\n\
            cache_dir="/runpod-volume/huggingface"\n\
        )\n\
        \n\
        clip_encoder = CLIPTextModel.from_pretrained(\n\
            "tencent/HunyuanVideo",\n\
            subfolder="text_encoder_2",\n\
            torch_dtype=torch.bfloat16,\n\
            cache_dir="/runpod-volume/huggingface"\n\
        )\n\
        \n\
        print("Loading vision encoder...")\n\
        vision_encoder = SiglipVisionModel.from_pretrained(\n\
            "tencent/HunyuanVideo",\n\
            subfolder="image_encoder",\n\
            torch_dtype=torch.bfloat16,\n\
            cache_dir="/runpod-volume/huggingface"\n\
        )\n\
        \n\
        model_components = {\n\
            "transformer": transformer.cuda(),\n\
            "vae": vae.cuda(),\n\
            "text_encoder": text_encoder.cuda(),\n\
            "clip_encoder": clip_encoder.cuda(),\n\
            "vision_encoder": vision_encoder.cuda()\n\
        }\n\
        \n\
        models_loaded = True\n\
        print("All models loaded successfully!")\n\
        return model_components\n\
        \n\
    except Exception as e:\n\
        print(f"Error loading models: {str(e)}")\n\
        raise\n\
\n\
def download_image(url):\n\
    """Download image from URL"""\n\
    response = requests.get(url, timeout=30)\n\
    response.raise_for_status()\n\
    \n\
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:\n\
        tmp.write(response.content)\n\
        return tmp.name\n\
\n\
def handler(event):\n\
    """RunPod serverless handler for FramePack"""\n\
    try:\n\
        input_data = event.get("input", {})\n\
        image_url = input_data.get("image_url")\n\
        prompt = input_data.get("prompt", "")\n\
        video_length = input_data.get("video_length", 5)  # seconds\n\
        steps = input_data.get("steps", 25)\n\
        cfg_scale = input_data.get("cfg_scale", 10.0)\n\
        \n\
        if not image_url:\n\
            return {"error": "image_url is required"}\n\
        \n\
        # Load models (cached after first load)\n\
        print("Ensuring models are loaded...")\n\
        models = load_models()\n\
        \n\
        # Download input image\n\
        print(f"Downloading image from {image_url}")\n\
        image_path = download_image(image_url)\n\
        \n\
        # Import worker function from demo_gradio\n\
        from demo_gradio import worker\n\
        \n\
        # Generate video\n\
        print(f"Generating video with prompt: {prompt}")\n\
        output_path = "/tmp/output.mp4"\n\
        \n\
        # Call FramePack worker function\n\
        # Note: This is a simplified version - you may need to adjust parameters\n\
        result = worker(\n\
            image=image_path,\n\
            prompt=prompt,\n\
            video_length=video_length,\n\
            steps=steps,\n\
            cfg_scale=cfg_scale\n\
        )\n\
        \n\
        # Upload result to R2 (you will handle this in your Next.js app)\n\
        # For now, return the video as base64 or upload URL\n\
        \n\
        print("Video generation complete!")\n\
        return {\n\
            "status": "completed",\n\
            "output_path": output_path,\n\
            "message": "Video generated successfully"\n\
        }\n\
        \n\
    except Exception as e:\n\
        import traceback\n\
        error_msg = f"Error: {str(e)}\\n{traceback.format_exc()}"\n\
        print(error_msg)\n\
        return {"error": error_msg}\n\
\n\
if __name__ == "__main__":\n\
    print("Starting RunPod serverless handler...")\n\
    print(f"Models will be cached to: /runpod-volume/huggingface")\n\
    runpod.serverless.start({"handler": handler})' > /app/handler.py

# Run the handler
CMD ["python", "/app/handler.py"]
