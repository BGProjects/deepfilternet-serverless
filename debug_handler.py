#!/usr/bin/env python3
"""
Debug wrapper for DeepFilterNet handler
Use this script to debug issues locally before deploying to RunPod
"""

import sys
import json
import traceback
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from handler import handler, main, load_processor
from utils import create_test_input, log_system_info

def debug_handler():
    """Debug the handler with various test scenarios"""
    
    print("🐛 DeepFilterNet Handler Debug Tool")
    print("=" * 50)
    
    # Check if test_input.json exists
    test_input_file = Path("test_input.json")
    
    if test_input_file.exists():
        print("📄 Using test_input.json")
        with open(test_input_file, 'r') as f:
            test_job = json.load(f)
    else:
        print("🔄 Creating synthetic test input")
        test_job = create_test_input(duration_s=3.0, sample_rate=48000)
    
    print(f"📝 Test input: {json.dumps(test_job, indent=2)}")
    print("-" * 50)
    
    try:
        # Log system info
        log_system_info()
        print("-" * 50)
        
        # Try to load processor first
        print("🚀 Testing processor loading...")
        try:
            processor = load_processor()
            print("✅ Processor loaded successfully")
        except Exception as e:
            print(f"❌ Processor loading failed: {e}")
            traceback.print_exc()
            return
        
        print("-" * 50)
        
        # Test handler
        print("🎯 Testing handler...")
        result = handler(test_job)
        
        if result.get('success', False):
            print("✅ Handler test PASSED")
            metadata = result.get('metadata', {})
            print(f"   Processing time: {metadata.get('total_processing_time_ms', 0):.1f}ms")
            print(f"   RTF: {metadata.get('rtf', 0):.3f}")
            print(f"   GPU used: {metadata.get('gpu_used', 'Unknown')}")
            
            # Check result size
            enhanced_audio = result.get('enhanced_audio_base64', '')
            if enhanced_audio:
                print(f"   Output size: {len(enhanced_audio)} chars (base64)")
            
        else:
            print("❌ Handler test FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()


def test_imports():
    """Test all imports"""
    
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime: {ort.__version__}")
        print(f"   Providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"❌ ONNX Runtime: {e}")
    
    try:
        import soundfile as sf
        print(f"✅ SoundFile: {sf.__version__}")
    except ImportError as e:
        print(f"❌ SoundFile: {e}")
    
    try:
        import runpod
        print(f"✅ RunPod: {runpod.__version__}")
    except ImportError as e:
        print(f"❌ RunPod: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--imports":
        test_imports()
    else:
        debug_handler()