#!/usr/bin/env python3
"""
Download DeepFilterNet3 ONNX models for serverless deployment.
This script downloads models from the original DeepFilterNet repository.
"""

import os
import sys
import requests
import tarfile
import tempfile
from pathlib import Path
from loguru import logger

def download_file(url: str, filename: str) -> bool:
    """Download file with progress tracking."""
    try:
        logger.info(f"Downloading {filename} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        sys.stdout.write(f'\r  Progress: {progress:.1f}%')
                        sys.stdout.flush()
        
        print()  # New line after progress
        logger.success(f"Downloaded {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False

def extract_models(tar_path: str, model_dir: str, model_type: str):
    """Extract ONNX models from tar file."""
    try:
        logger.info(f"Extracting {model_type} models to {model_dir}")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Extract only .onnx and .ini files
            for member in tar.getmembers():
                if member.name.endswith(('.onnx', '.ini')):
                    # Extract to model directory
                    member.name = os.path.basename(member.name)
                    tar.extract(member, model_dir)
                    
        logger.success(f"Extracted {model_type} models")
        
    except Exception as e:
        logger.error(f"Failed to extract {model_type} models: {e}")

def main():
    """Download and setup ONNX models."""
    logger.info("🎵 Downloading DeepFilterNet3 ONNX models...")
    
    # Model URLs from DeepFilterNet releases
    models = {
        'normal': 'https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_onnx.tar.gz',
        'lowlatency': 'https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3_ll_onnx.tar.gz'
    }
    
    # Create model directories
    base_dir = Path('/app/models' if os.path.exists('/app') else 'models')
    base_dir.mkdir(exist_ok=True)
    
    for model_type, url in models.items():
        model_dir = base_dir / model_type
        model_dir.mkdir(exist_ok=True)
        
        # Skip if models already exist
        if (model_dir / 'enc.onnx').exists():
            logger.info(f"{model_type} models already exist, skipping download")
            continue
            
        # Download to temporary file
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            if download_file(url, tmp_file.name):
                extract_models(tmp_file.name, str(model_dir), model_type)
                os.unlink(tmp_file.name)
            else:
                logger.error(f"Failed to download {model_type} models")
                return False
    
    logger.success("🎉 All models downloaded successfully!")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)