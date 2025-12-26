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
# This is critical - HuggingFace downloads to temp first!
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/huggingface/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/runpod-volume/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/runpod-volume/huggingface/datasets"
os.environ["TORCH_HOME"] = "/runpod-volume/torch"
os.environ["TMPDIR"] = "/runpod-volume/tmp"
os.environ["TEMP"] = "/runpod-volume/tmp"
os.environ["TMP"] = "/runpod-volume/tmp"

print(f"Environment variables set:")
print(f"  HF_HOME={os.environ['HF_HOME']}")
print(f"  HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")
print(f"  TMPDIR={os.environ['TMPDIR']}")

# Add FramePack to path
sys.path.insert(0, "/app")

# Import FramePack's demo_gradio module
# This will handle all the model loading and processing
import demo_gradio

def download_image(url):
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Open as PIL Image
        image = Image.open(tmp_path)
        return image
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")

def handler(event):
    """RunPod serverless handler for FramePack"""
    try:
        print("=" * 50)
        print("Starting FramePack video generation")
        print("=" * 50)

        input_data = event.get("input", {})
        image_url = input_data.get("image_url")
        prompt = input_data.get("prompt", "")

        # FramePack parameters with defaults
        n_prompt = input_data.get("n_prompt", "")  # negative prompt
        seed = input_data.get("seed", 12345)
        total_second_length = input_data.get("video_length", 5)  # seconds
        latent_window_size = input_data.get("latent_window_size", 3)
        steps = input_data.get("steps", 25)
        cfg = input_data.get("cfg_scale", 10.0)
        gs = input_data.get("gs", 1.0)  # guidance scale
        rs = input_data.get("rs", 0.5)  # random scale
        gpu_memory_preservation = input_data.get("gpu_memory_preservation", 0.0)
        use_teacache = input_data.get("use_teacache", False)
        mp4_crf = input_data.get("mp4_crf", 18)

        if not image_url:
            return {"error": "image_url is required"}

        print(f"Downloading image from: {image_url}")
        input_image = download_image(image_url)
        print(f"Image downloaded: {input_image.size}")

        print(f"Prompt: {prompt}")
        print(f"Parameters: steps={steps}, cfg={cfg}, length={total_second_length}s")

        # Call FramePack's process function
        # This handles model loading, video generation, everything
        print("Starting FramePack processing...")

        # The process function is a generator that yields progress updates
        # We need to iterate through it to get the final result
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

        # Iterate through the generator to completion
        output_video_path = None
        for result in result_generator:
            # result is a tuple: (output_filename, preview_image, description, progress_html, ...)
            if result and len(result) > 0:
                output_video_path = result[0]
                if result[2]:  # description
                    print(f"Progress: {result[2]}")

        if not output_video_path or not os.path.exists(output_video_path):
            return {"error": "Video generation failed - no output file created"}

        print(f"Video generated successfully: {output_video_path}")

        # Read the video file
        with open(output_video_path, 'rb') as f:
            video_data = f.read()

        # For now, we'll return the base64-encoded video
        # In production, you'd upload this to R2
        import base64
        video_base64 = base64.b64encode(video_data).decode('utf-8')

        print("=" * 50)
        print("Video generation complete!")
        print("=" * 50)

        return {
            "status": "completed",
            "video_base64": video_base64,
            "video_size_bytes": len(video_data),
            "message": "Video generated successfully"
        }

    except Exception as e:
        error_msg = f"Error in handler: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    print("Starting RunPod serverless handler for FramePack...")
    print(f"Models will be cached to: /runpod-volume/huggingface")
    runpod.serverless.start({"handler": handler})
