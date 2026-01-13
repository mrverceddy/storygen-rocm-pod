# StoryGen GPU Pod - ROCm/AMD Version

GPU server for StoryGen pipeline running on AMD GPUs (MI300X, MI250X, etc.) with ROCm.

## Why ROCm/AMD?

- **MI300X**: 192GB VRAM at ~$1.99/hr on RunPod
- Load multiple models simultaneously without swapping
- Competitive performance for inference workloads

## Quick Start

### 1. Create RunPod Pod

1. Go to RunPod and create a new GPU Pod
2. Select an AMD GPU (MI300X recommended)
3. Use a ROCm-compatible template (e.g., `runpod/pytorch:2.1.0-py3.10-rocm5.7`)
4. Set at least 50GB disk space for models

### 2. Setup the Pod

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/storygen-rocm-pod.git /workspace/storygen-rocm-pod

# Run setup (downloads ~40GB of models)
cd /workspace/storygen-rocm-pod
chmod +x pod_setup.sh
./pod_setup.sh
```

### 3. Start the Server

```bash
python /workspace/storygen-rocm-pod/pod_server.py
```

Server runs on port 8000. Access via your pod's public IP.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate_image` | POST | Flux text-to-image |
| `/generate_video` | POST | Wan 2.2 image-to-video |
| `/generate_voice` | POST | Parler-TTS voice generation |
| `/synthesize_voice` | POST | Fish Speech voice cloning |
| `/train_lora` | POST | Kohya LoRA training |
| `/health` | GET | GPU status and loaded models |
| `/clear_vram` | POST | Unload models from GPU |
| `/download/{filename}` | GET | Download generated files |

## Health Check

```bash
curl http://POD_IP:8000/health
```

Response:
```json
{
  "status": "ok",
  "gpu": "AMD Instinct MI300X",
  "vram_gb": 192.0,
  "backend": "ROCm/HIP",
  "is_rocm": true,
  "models_loaded": ["qwen_image"]
}
```

## ROCm vs CUDA Differences

| Feature | CUDA Version | ROCm Version |
|---------|--------------|--------------|
| PyTorch | cu124 | rocm6.0 |
| bitsandbytes | Yes | No (not supported) |
| Mixed precision | bf16 | fp16 (safer) |
| Flash Attention | Full support | Experimental |

## MI300X Advantages

With 192GB VRAM, the server can:
- Load all models simultaneously (no `/clear_vram` needed)
- Run larger batch sizes
- Handle higher resolution outputs

The server automatically detects large VRAM and keeps models loaded.

## Troubleshooting

### ROCm not detected
```bash
rocm-smi --showproductname
```
If this fails, ensure you're using a ROCm-compatible template.

### Model loading issues
Some models may need fp16 instead of bf16 on ROCm:
```python
torch_dtype=torch.float16  # instead of torch.bfloat16
```

### Kohya training issues
Kohya sd-scripts has limited ROCm support. If training fails:
1. Ensure using `--mixed_precision fp16`
2. Try reducing batch size
3. Check ROCm version compatibility

## Files

- `pod_setup.sh` - One-time setup script
- `pod_server.py` - FastAPI GPU server
- `requirements.txt` - Python dependencies

## License

MIT
