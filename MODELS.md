# Model Files

## Download Models

Due to GitHub repository size limits, ONNX model files are distributed separately:

### Option 1: Download from Original DeepFilterNet
```bash
# Download models from original repository  
mkdir -p models/normal models/lowlatency

# Download normal models
wget -O models/normal/enc.onnx https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_onnx.tar.gz
wget -O models/normal/erb_dec.onnx https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_onnx.tar.gz  
wget -O models/normal/df_dec.onnx https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_onnx.tar.gz

# Download low-latency models
wget -O models/lowlatency/enc.onnx https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_ll_onnx.tar.gz
wget -O models/lowlatency/erb_dec.onnx https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_ll_onnx.tar.gz
wget -O models/lowlatency/df_dec.onnx https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_ll_onnx.tar.gz
```

### Option 2: Use GitHub Release (Coming Soon)
We will upload model files as GitHub release assets.

## Model Structure
```
models/
├── normal/           # Standard DeepFilterNet3 
│   ├── config.ini   # Model configuration
│   ├── enc.onnx     # Encoder model (1.9MB)
│   ├── erb_dec.onnx # ERB decoder (3.2MB)  
│   └── df_dec.onnx  # DF decoder (3.2MB)
└── lowlatency/      # Low-latency version
    ├── config.ini   # Model configuration
    ├── enc.onnx     # Encoder model (6.7MB)
    ├── erb_dec.onnx # ERB decoder (13MB)
    └── df_dec.onnx  # DF decoder (19MB)
```

## Usage in Docker
The Dockerfile will automatically download models during build:

```dockerfile
RUN python /app/scripts/download_models.py
```