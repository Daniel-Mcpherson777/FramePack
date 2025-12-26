import runpod
import sys
import os
import tempfile
import requests
import traceback
from PIL import Image
from pathlib import Path

# Create cache directories on network volume FIRST
cache_dirs = [
    "/runpod-volume/huggingface",
    "/runpod-volume/huggingface/hub",
    "/runpod-volume/huggingface/transformers",
    "/runpod-volume/huggingface/datasets",
    "/runpod-volume/torch",
    "/runpod-volume/tmp",
    "/runpod-volume/hf_download"
]

for cache_dir in cache_dirs:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"Created cache directory: {cache_dir}")

# Create symlink to redirect hardcoded /app/hf_download to network volume
hf_download_symlink = "/app/hf_download"
hf_download_target = "/runpod-volume/hf_download"

# Remove existing symlink/directory if it exists
if os.path.exists(hf_download_symlink) or os.path.islink(hf_download_symlink):
    if os.path.islink(hf_download_symlink):
        os.unlink(hf_download_symlink)
        print(f"Removed old symlink: {hf_download_symlink}")
    elif os.path.isdir(hf_download_symlink):
        import shutil
        shutil.rmtree(hf_download_symlink)
        print(f"Removed old directory: {hf_download_symlink}")

# Create the symlink
os.symlink(hf_download_target, hf_download_symlink)
print(f"Created symlink: {hf_download_symlink} -> {hf_download_target}")

# Set ALL cache and temp directories to network volume
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/huggingface/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/runpod-volume/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/runpod-volume/huggingface/datasets"
os.environ["TORCH_HOME"] = "/runpod-volume/torch"
os.environ["TMPDIR"] = "/runpod-volume/tmp"
os.environ["TEMP"] = "/runpod-volume/tmp"
os.environ["TMP"] = "/runpod-volume/tmp"

print(f"Starting RunPod Handler...")
print(f"Environment variables set:")
print(f"  HF_HOME={os.environ['HF_HOME']}")
print(f"  HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")
print(f"  TMPDIR={os.environ['TMPDIR']}")

# Add FramePack to path
sys.path.insert(0, "/app")

def download_image(url):
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        image = Image.open(tmp_path)
        return image
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")

def handler(event):
    """RunPod serverless handler for FramePack"""
    try:
        print("=" * 60)
        print("FRAMEPACK JOB STARTED")
        print("=" * 60)

        input_data = event.get("input", {})
        image_url = input_data.get("image_url")
        prompt = input_data.get("prompt", "")

        if not image_url:
            return {"error": "image_url is required"}

        print(f"Image URL: {image_url}")
        print(f"Prompt: {prompt}")
        print("Downloading input image...")

        input_image = download_image(image_url)
        print(f"Image downloaded successfully: {input_image.size}")

        # Convert PIL Image to numpy array for inference
        import numpy as np
        input_image_np = np.array(input_image)

        # Import clean inference module (NO Gradio!)
        print("Loading FramePack inference (this may take a few minutes on first run)...")
        import inference

        print("Starting video generation...")

        # Call FramePack with parameters (optimized for quality)
        n_prompt = ""
        seed = 12345
        total_second_length = 0.6  # Generate 18 frames (0.6 * 30fps) for 3-second video @ 6fps
        latent_window_size = 3
        steps = 40  # Increased from 25 for better quality
        cfg = 10.0
        gs = 1.0
        rs = 0.5
        gpu_memory_preservation = 3.0  # Reduced from 6GB - H100 has plenty of VRAM
        use_teacache = False
        mp4_crf = 12  # Lower = better quality (was 18)
        fps = 6  # 6fps playback (18 frames / 6fps = 3 seconds)

        # Call the clean inference function (returns path directly, no generator)
        output_video_path = inference.generate_video(
            input_image=input_image_np,
            prompt=prompt,
            n_prompt=n_prompt,
            seed=seed,
            total_second_length=total_second_length,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            mp4_crf=mp4_crf,
            fps=fps
        )

        if not output_video_path or not os.path.exists(output_video_path):
            return {"error": "Video generation failed - no output file created"}

        print(f"Video generated: {output_video_path}")

        # Upload video to Cloudflare R2
        import boto3
        from datetime import datetime

        print("Uploading video to Cloudflare R2...")

        # R2 credentials from environment (set in Dockerfile or RunPod secrets)
        r2_account_id = os.environ.get("R2_ACCOUNT_ID")
        r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
        r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        r2_bucket = os.environ.get("R2_BUCKET_NAME", "graphicsgod-storage")
        r2_public_url = os.environ.get("R2_PUBLIC_URL", "https://pub-2edbe636fe384a5b881ec1342972f472.r2.dev")

        # Create S3 client for R2
        s3 = boto3.client(
            's3',
            endpoint_url=f'https://{r2_account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            region_name='auto'
        )

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"videos/generated_{timestamp}.mp4"

        # Upload to R2
        with open(output_video_path, 'rb') as f:
            s3.upload_fileobj(
                f,
                r2_bucket,
                filename,
                ExtraArgs={'ContentType': 'video/mp4'}
            )

        # Construct public URL
        video_url = f"{r2_public_url}/{filename}"

        # Get file size
        video_size = os.path.getsize(output_video_path)

        print("=" * 60)
        print("VIDEO GENERATION COMPLETE!")
        print(f"Video size: {video_size} bytes")
        print(f"Video URL: {video_url}")
        print("=" * 60)

        return {
            "status": "completed",
            "video_url": video_url,
            "video_size_bytes": video_size,
            "message": "Video generated and uploaded successfully"
        }

    except Exception as e:
        error_msg = f"Handler error: {str(e)}\n{traceback.format_exc()}"
        print("=" * 60)
        print("ERROR IN HANDLER")
        print(error_msg)
        print("=" * 60)
        return {"error": error_msg}

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING RUNPOD SERVERLESS HANDLER")
    print("Models will cache to: /runpod-volume/huggingface")
    print("=" * 60)
    runpod.serverless.start({"handler": handler})
