# DeepFilterNet3 RunPod Deployment Notes

## Critical Fixes Applied - August 2025

### 🚨 Issues Resolved:

1. **'tuple' object has no attribute 'split' Error**
   - **Problem**: ONNX provider logging tried to split tuple objects
   - **Fix**: Added type checking in `onnx_inference.py:47-60`
   - **Location**: `src/onnx_inference.py`

2. **CUDA Library Compatibility**
   - **Problem**: ONNX Runtime 1.16.3 looking for CUDA 11.x libraries in CUDA 12.4 environment
   - **Fix**: Upgraded to ONNX Runtime 1.18.1 with CUDA 12.x support
   - **Changes**: Updated Dockerfile and requirements.txt

3. **RTX 5090 sm_120 PyTorch Incompatibility**
   - **Problem**: PyTorch 2.1.2 doesn't support CUDA capability sm_120
   - **Fix**: Switched to PyTorch nightly builds with CUDA 12.8 support
   - **Changes**: Added PyTorch nightly installation in Dockerfile

4. **ONNX Runtime CUDA Provider Initialization**
   - **Problem**: ONNX Runtime couldn't find CUDA libraries
   - **Fix**: Added `onnxruntime.preload_dlls()` call in handler.py
   - **Improvement**: Import PyTorch before ONNX Runtime to load CUDA libs

## Deployment Requirements:

### Hardware:
- RTX 5090 (primary target)
- RTX 5080, RTX 4090 (compatible)
- 16GB+ GPU memory recommended

### Software Stack:
- CUDA 12.4+ (container includes)
- PyTorch 2.8.0+ nightly
- ONNX Runtime 1.18.1+ with CUDA 12.x
- Python 3.10

### RunPod Settings:
```json
{
  "gpu": "RTX 5090",
  "memory": "16GB",
  "storage": "50GB",
  "cuda_version": "12.4",
  "environment": {
    "CUDA_MEMORY_FRACTION": "0.7",
    "GPU_MEMORY_UTILIZATION": "0.7",
    "ENABLE_TENSORRT": "false"
  }
}
```

## Build Command:
```bash
docker build -t deepfilternet3-rtx5090:v2.0 .
```

## Testing:
```bash
# Local test
python debug_handler.py

# RunPod test  
curl -X POST https://api.runpod.ai/v2/ENDPOINT/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer API_KEY' \
  -d @test_input.json
```

## Performance Expectations:
- RTX 5090: ~0.02-0.05 RTF (20-50x real-time)
- RTX 4090: ~0.03-0.07 RTF (15-30x real-time)
- CPU Fallback: ~2.0-5.0 RTF (0.2-0.5x real-time)

## Troubleshooting:
1. Check GPU detection: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check ONNX providers: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
4. Test handler locally: `python debug_handler.py`

## Version History:
- v1.0.0: Initial RTX 4090 support
- v2.0.0: RTX 5090 support, CUDA 12.8, PyTorch nightly, critical bug fixes