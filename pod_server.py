"""
StoryGen GPU Server for RunPod Pods - ROCm/AMD Version.

For AMD MI300X and other ROCm-compatible GPUs.

Run with: python pod_server.py
Access at: http://POD_IP:8000

All GPU tasks available via REST API.
"""

import base64
import gc
import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Config
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
OUTPUT_DIR = Path("/workspace/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ROCm uses the same torch.cuda API (HIP translates calls)
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="StoryGen GPU Server (ROCm)")

# Global model cache
_models = {}


def clear_vram():
    """Unload all models to free VRAM."""
    global _models
    for name in list(_models.keys()):
        del _models[name]
    _models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # ROCm-specific: also try to clear HIP cache
        if IS_ROCM:
            try:
                torch.cuda.synchronize()
            except:
                pass


def get_gpu_info():
    """Get GPU info compatible with both CUDA and ROCm."""
    if not torch.cuda.is_available():
        return {"name": "none", "vram_gb": 0, "backend": "cpu"}

    return {
        "name": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "backend": "ROCm/HIP" if IS_ROCM else "CUDA",
        "vram_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9,
    }


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class GenerateImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    num_steps: int = 28
    guidance: float = 3.5
    seed: int = -1


class EditImageRequest(BaseModel):
    image_base64: str
    prompt: str


class GenerateVideoRequest(BaseModel):
    start_frame_base64: str
    prompt: str
    end_frame_base64: Optional[str] = None
    num_frames: int = 81
    width: int = 1280
    height: int = 720
    num_steps: int = 30
    guidance: float = 5.0
    seed: int = -1


class GenerateVoiceRequest(BaseModel):
    text: str
    description: str


class SynthesizeVoiceRequest(BaseModel):
    reference_audio_base64: str
    text: str


class TrainLoRARequest(BaseModel):
    dataset_url: str
    lora_name: str
    num_epochs: int = 10
    learning_rate: float = 1e-4
    network_rank: int = 32
    network_alpha: int = 16
    resolution: int = 1024


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def load_qwen_image():
    if "qwen_image" not in _models:
        clear_vram()
        print(f"Loading Qwen-Image-2512 on {DEVICE} ({'ROCm' if IS_ROCM else 'CUDA'})...")
        from diffusers import QwenImagePipeline

        # Use float16 for ROCm compatibility (bfloat16 support varies)
        dtype = torch.float16 if IS_ROCM else torch.bfloat16

        _models["qwen_image"] = QwenImagePipeline.from_pretrained(
            MODEL_DIR / "qwen-image",
            torch_dtype=dtype,
        )
        # For MI300X with 192GB, we can load directly to GPU
        if get_gpu_info()["vram_gb"] > 100:
            _models["qwen_image"] = _models["qwen_image"].to(DEVICE)
        else:
            _models["qwen_image"].enable_model_cpu_offload()
    return _models["qwen_image"]


@app.post("/generate_image")
async def generate_image(req: GenerateImageRequest):
    pipe = load_qwen_image()

    generator = torch.Generator(DEVICE).manual_seed(req.seed) if req.seed >= 0 else None

    image = pipe(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_steps,
        guidance_scale=req.guidance,
        generator=generator,
    ).images[0]

    # Save and return
    path = OUTPUT_DIR / f"image_{os.urandom(4).hex()}.png"
    image.save(path)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode(),
        "path": str(path),
    }


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def load_wan():
    if "wan" not in _models:
        clear_vram()
        print(f"Loading Wan 2.1 FLF2V on {DEVICE} ({'ROCm' if IS_ROCM else 'CUDA'})...")
        from diffusers import WanFLFToVideoPipeline

        # Use float16 for ROCm compatibility
        dtype = torch.float16 if IS_ROCM else torch.bfloat16

        _models["wan"] = WanFLFToVideoPipeline.from_pretrained(
            MODEL_DIR / "wan-flf2v",
            torch_dtype=dtype,
        )
        # For MI300X with 192GB, we can load directly to GPU
        if get_gpu_info()["vram_gb"] > 100:
            _models["wan"] = _models["wan"].to(DEVICE)
        else:
            _models["wan"].enable_model_cpu_offload()
    return _models["wan"]


@app.post("/generate_video")
async def generate_video(req: GenerateVideoRequest):
    from PIL import Image
    from diffusers.utils import export_to_video

    pipe = load_wan()

    # Decode start frame
    start_img = Image.open(io.BytesIO(base64.b64decode(req.start_frame_base64)))
    start_img = start_img.resize((req.width, req.height))

    # Decode end frame if provided
    end_img = None
    if req.end_frame_base64:
        end_img = Image.open(io.BytesIO(base64.b64decode(req.end_frame_base64)))
        end_img = end_img.resize((req.width, req.height))

    generator = torch.Generator(DEVICE).manual_seed(req.seed) if req.seed >= 0 else None

    # FLF2V uses first_image and last_image parameters
    output = pipe(
        first_image=start_img,
        last_image=end_img,
        prompt=req.prompt,
        num_frames=req.num_frames,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_steps,
        guidance_scale=req.guidance,
        generator=generator,
    )

    frames = output.frames[0]

    path = OUTPUT_DIR / f"video_{os.urandom(4).hex()}.mp4"
    export_to_video(frames, str(path), fps=16)

    with open(path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    return {
        "video_base64": video_b64,
        "path": str(path),
        "duration_seconds": req.num_frames / 16.0,
    }


# =============================================================================
# VOICE GENERATION
# =============================================================================

def load_parler():
    if "parler" not in _models:
        clear_vram()
        print(f"Loading Parler-TTS on {DEVICE}...")
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        _models["parler"] = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_DIR / "parler-tts",
            torch_dtype=torch.float16,
        ).to(DEVICE)
        _models["parler_tok"] = AutoTokenizer.from_pretrained(MODEL_DIR / "parler-tts")
    return _models["parler"], _models["parler_tok"]


@app.post("/generate_voice")
async def generate_voice(req: GenerateVoiceRequest):
    import torchaudio

    model, tokenizer = load_parler()

    input_ids = tokenizer(req.description, return_tensors="pt").input_ids.to(DEVICE)
    prompt_ids = tokenizer(req.text, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        audio = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)

    audio_np = audio.cpu().numpy().squeeze()
    sample_rate = model.config.sampling_rate

    path = OUTPUT_DIR / f"voice_{os.urandom(4).hex()}.wav"
    torchaudio.save(str(path), torch.tensor(audio_np).unsqueeze(0), sample_rate)

    with open(path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    return {
        "audio_base64": audio_b64,
        "path": str(path),
        "duration_seconds": len(audio_np) / sample_rate,
    }


# =============================================================================
# VOICE SYNTHESIS (Fish Speech)
# =============================================================================

@app.post("/synthesize_voice")
async def synthesize_voice(req: SynthesizeVoiceRequest):
    clear_vram()  # Fish Speech runs as subprocess

    # Save reference audio
    ref_path = OUTPUT_DIR / f"ref_{os.urandom(4).hex()}.wav"
    ref_path.write_bytes(base64.b64decode(req.reference_audio_base64))

    out_path = OUTPUT_DIR / f"synth_{os.urandom(4).hex()}.wav"

    cmd = [
        "python", "-m", "fish_speech.tools.inference",
        "--checkpoint", str(MODEL_DIR / "fish-speech"),
        "--reference", str(ref_path),
        "--text", req.text,
        "--output", str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if not out_path.exists():
        raise HTTPException(500, f"Synthesis failed: {result.stderr}")

    with open(out_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    return {
        "audio_base64": audio_b64,
        "path": str(out_path),
    }


# =============================================================================
# LORA TRAINING
# =============================================================================

@app.post("/train_lora")
async def train_lora(req: TrainLoRARequest):
    import zipfile
    from urllib.request import urlretrieve

    clear_vram()  # Need all VRAM for training

    # Setup dirs
    dataset_dir = Path("/tmp/dataset")
    output_dir = Path("/workspace/loras")
    dataset_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Download dataset
    zip_path = dataset_dir / "dataset.zip"
    urlretrieve(req.dataset_url, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dataset_dir / "images")

    # Run Kohya training
    # Note: Kohya has limited ROCm support, using fp16 instead of bf16 for compatibility
    cmd = [
        "accelerate", "launch",
        "/workspace/sd-scripts/flux_train_network.py",
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-dev",
        "--train_data_dir", str(dataset_dir / "images"),
        "--output_dir", str(output_dir),
        "--output_name", req.lora_name,
        "--max_train_epochs", str(req.num_epochs),
        "--learning_rate", str(req.learning_rate),
        "--network_dim", str(req.network_rank),
        "--network_alpha", str(req.network_alpha),
        "--resolution", f"{req.resolution},{req.resolution}",
        "--mixed_precision", "fp16",  # Use fp16 for ROCm compatibility
        "--network_module", "networks.lora",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    lora_path = output_dir / f"{req.lora_name}.safetensors"

    if not lora_path.exists():
        raise HTTPException(500, f"Training failed: {result.stderr[-500:]}")

    return {
        "path": str(lora_path),
        "size_mb": lora_path.stat().st_size / (1024*1024),
        "log": result.stdout[-1000:],
    }


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    gpu_info = get_gpu_info()
    return {
        "status": "ok",
        "gpu": gpu_info["name"],
        "vram_gb": gpu_info["vram_gb"],
        "vram_free_gb": gpu_info.get("vram_free_gb", 0),
        "backend": gpu_info["backend"],
        "is_rocm": IS_ROCM,
        "models_loaded": list(_models.keys()),
    }


@app.post("/clear_vram")
async def clear():
    clear_vram()
    gpu_info = get_gpu_info()
    return {
        "status": "cleared",
        "vram_free_gb": gpu_info.get("vram_free_gb", 0),
    }


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@app.on_event("startup")
async def startup_preload():
    """Preload the image model on startup to avoid timeout on first request."""
    gpu_info = get_gpu_info()
    print(f"Starting on {gpu_info['name']} ({gpu_info['vram_gb']:.1f}GB VRAM) - {gpu_info['backend']}")

    # With 192GB VRAM on MI300X, we can preload multiple models
    if gpu_info["vram_gb"] > 100:
        print("Large VRAM detected - preloading multiple models...")
        try:
            load_qwen_image()
            print("Qwen-Image preloaded!")
            # Could also preload Wan and Parler with 192GB
        except Exception as e:
            print(f"Warning: Failed to preload model: {e}")
    else:
        print("Preloading Qwen-Image-2512 for fast first request...")
        try:
            load_qwen_image()
            print("Model preloaded and ready!")
        except Exception as e:
            print(f"Warning: Failed to preload model: {e}")


if __name__ == "__main__":
    print("Starting StoryGen GPU Server (ROCm/AMD)...")
    print(f"Models dir: {MODEL_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"ROCm detected: {IS_ROCM}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
