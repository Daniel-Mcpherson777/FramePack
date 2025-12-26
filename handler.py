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
    "/runpod-volume/tmp"
]

for cache_dir in cache_dirs:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"Created cache directory: {cache_dir}")

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

        # Import demo_gradio INSIDE the handler to avoid auto-launch
        print("Loading FramePack (this may take a few minutes on first run)...")
        import demo_gradio

        print("Calling FramePack process function...")

        # Call FramePack with parameters
        n_prompt = ""
        seed = 12345
        total_second_length = 5
        latent_window_size = 3
        steps = 25
        cfg = 10.0
        gs = 1.0
        rs = 0.5
        gpu_memory_preservation = 0.0
        use_teacache = False
        mp4_crf = 18

        # Process returns a generator
        result_generator = demo_gradio.process(
            input_image=input_image,
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
            mp4_crf=mp4_crf
        )

        print("Processing video generation...")

        # Iterate through generator to completion
        output_video_path = None
        for i, result in enumerate(result_generator):
            if result and len(result) > 0:
                output_video_path = result[0]
                if len(result) > 2 and result[2]:
                    print(f"Progress update {i}: {result[2]}")

        if not output_video_path or not os.path.exists(output_video_path):
            return {"error": "Video generation failed - no output file created"}

        print(f"Video generated: {output_video_path}")

        # Read video file
        with open(output_video_path, 'rb') as f:
            video_data = f.read()

        # Return base64 encoded video
        import base64
        video_base64 = base64.b64encode(video_data).decode('utf-8')

        print("=" * 60)
        print("VIDEO GENERATION COMPLETE!")
        print(f"Video size: {len(video_data)} bytes")
        print("=" * 60)

        return {
            "status": "completed",
            "video_base64": video_base64,
            "video_size_bytes": len(video_data),
            "message": "Video generated successfully"
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
