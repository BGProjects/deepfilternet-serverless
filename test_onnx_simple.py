#!/usr/bin/env python3
"""
Simplified ONNX test script for DeepFilterNet3
"""

import os
import sys
import tempfile
import requests
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required packages are available"""
    try:
        import numpy as np
        print("✓ numpy imported")
        
        import requests
        print("✓ requests imported")
        
        # Test if we can download the file
        url = "https://cozmo.com.tr/videotest.wav"
        response = requests.head(url, timeout=10)
        print(f"✓ Audio URL accessible: {response.status_code}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def download_test_file():
    """Download the test audio file"""
    url = "https://cozmo.com.tr/videotest.wav"
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        print(f"Downloading from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
            total_size += len(chunk)
        
        print(f"Downloaded {total_size} bytes to {tmp_file.name}")
        return tmp_file.name

def check_models():
    """Check if ONNX models exist"""
    models_dir = Path(__file__).parent / "models"
    
    dfn3_onnx = models_dir / "DeepFilterNet3_onnx.tar.gz"
    dfn3_ll_onnx = models_dir / "DeepFilterNet3_ll_onnx.tar.gz"
    
    print(f"Checking models in: {models_dir}")
    print(f"DFN3 ONNX: {dfn3_onnx.exists()} ({dfn3_onnx})")
    print(f"DFN3 LL ONNX: {dfn3_ll_onnx.exists()} ({dfn3_ll_onnx})")
    
    if dfn3_onnx.exists():
        print(f"DFN3 ONNX size: {dfn3_onnx.stat().st_size / 1024 / 1024:.1f} MB")
    
    if dfn3_ll_onnx.exists():
        print(f"DFN3 LL ONNX size: {dfn3_ll_onnx.stat().st_size / 1024 / 1024:.1f} MB")
    
    return dfn3_onnx.exists() or dfn3_ll_onnx.exists()

def main():
    """Main test function"""
    print("=== DeepFilterNet3 ONNX Test Suite ===\n")
    
    # Test 1: Check imports
    print("1. Testing imports...")
    if not test_imports():
        print("Please install required packages:")
        print("pip install onnxruntime soundfile requests scipy librosa loguru")
        return False
    print()
    
    # Test 2: Check models
    print("2. Checking ONNX models...")
    if not check_models():
        print("ONNX models not found. Please ensure they are extracted.")
        return False
    print()
    
    # Test 3: Download test file
    print("3. Testing audio download...")
    try:
        audio_file = download_test_file()
        print(f"✓ Audio file downloaded: {audio_file}")
        
        # Clean up
        os.unlink(audio_file)
        print("✓ Test file cleaned up")
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False
    print()
    
    print("✓ All tests passed! Ready to run ONNX processing.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)