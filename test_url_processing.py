#!/usr/bin/env python3
"""
Test script for URL-based audio processing
Tests downloading from https://cozmo.com.tr/videotest.wav
"""

import sys
import time
sys.path.append('src')

from handler import handler

def test_url_processing():
    """Test processing audio from URL"""
    
    # Test data with your specific URL
    test_job = {
        "input": {
            "audio_url": "https://cozmo.com.tr/videotest.wav",
            "sample_rate": 48000,
            "return_metadata": True
        }
    }
    
    print("🎵 Testing URL-based audio processing...")
    print(f"URL: {test_job['input']['audio_url']}")
    
    start_time = time.time()
    
    try:
        result = handler(test_job)
        
        total_time = time.time() - start_time
        
        if result.get("success"):
            print("✅ Processing successful!")
            
            # Display timing information
            metadata = result.get("metadata", {})
            print(f"\n📊 Timing Breakdown:")
            print(f"  Total time: {metadata.get('total_processing_time_ms', 0):.1f}ms")
            print(f"  Download time: {metadata.get('download_time_ms', 0):.1f}ms")
            print(f"  Inference time: {metadata.get('inference_time_ms', 0):.1f}ms")
            print(f"  Encoding time: {metadata.get('encoding_time_ms', 0):.1f}ms")
            print(f"  RTF: {metadata.get('rtf', 'N/A')}")
            print(f"  Input duration: {metadata.get('input_duration_s', 'N/A')}s")
            print(f"  GPU: {metadata.get('gpu_used', 'N/A')}")
            print(f"  Input source: {metadata.get('input_source', 'N/A')}")
            
            # Check output
            enhanced_audio = result.get("enhanced_audio_base64", "")
            print(f"\n📤 Output:")
            print(f"  Enhanced audio size: {len(enhanced_audio)} chars (base64)")
            
        else:
            print("❌ Processing failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_url_processing()