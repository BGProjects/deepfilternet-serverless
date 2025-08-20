#!/usr/bin/env python3
"""
DeepFilterNet3 Serverless Handler for RunPod
Optimized for RTX 4090 GPU deployment
"""

import os
import sys
import time
import base64
import tempfile
import requests
from typing import Dict, Any, Optional
import traceback

import runpod
import torch
import numpy as np
import soundfile as sf
from loguru import logger

# Import our custom modules
from audio_processor import DeepFilterNetProcessor
from utils import setup_logging, validate_input, handle_error

# Global processor instance (loaded once per container)
processor: Optional[DeepFilterNetProcessor] = None

def download_audio_from_url(url: str, timeout: int = 30) -> tuple[bytes, float]:
    """
    Download audio file from URL with detailed timing.
    
    Returns:
        tuple: (audio_data, download_time_seconds)
    """
    download_start = time.time()
    
    try:
        logger.info(f"📥 Downloading audio from URL: {url}")
        
        # Add proper headers for audio download
        headers = {
            'User-Agent': 'DeepFilterNet3-Serverless/1.0',
            'Accept': 'audio/*',
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if response is actually audio
        content_type = response.headers.get('content-type', '').lower()
        if not any(audio_type in content_type for audio_type in ['audio/', 'application/octet-stream']):
            logger.warning(f"Unexpected content type: {content_type}")
        
        # Get file size for logging
        content_length = response.headers.get('content-length')
        if content_length:
            file_size_mb = int(content_length) / (1024 * 1024)
            logger.info(f"📦 File size: {file_size_mb:.2f} MB")
        
        # Download the audio data
        audio_data = response.content
        download_time = time.time() - download_start
        
        logger.info(f"✅ Downloaded {len(audio_data)} bytes in {download_time:.2f}s")
        
        return audio_data, download_time
        
    except requests.exceptions.Timeout:
        raise ValueError(f"Download timeout after {timeout}s")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download audio: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error during download: {str(e)}")


def load_processor():
    """Load the DeepFilterNet processor (called once per container)"""
    global processor
    
    if processor is None:
        logger.info("Loading DeepFilterNet3 processor...")
        start_time = time.time()
        
        try:
            # Use normal model by default, LL model available but not used
            model_path = "/app/models/normal"
            processor = DeepFilterNetProcessor(
                model_path=model_path,
                use_gpu=True,
                optimize_for_rtx4090=True
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Processor loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to load processor: {e}")
            raise
    
    return processor


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "audio_base64": "base64_encoded_wav_data",  # Option 1: Base64 audio data
            "audio_url": "https://example.com/audio.wav", # Option 2: URL to audio file
            "sample_rate": 48000,                       # Optional, default: 48000
            "output_format": "wav",                     # Optional, default: "wav"
            "attenuation_limit_db": null,               # Optional noise attenuation limit
            "return_metadata": true                     # Optional, return processing stats
        }
    }
    
    Returns:
    {
        "enhanced_audio_base64": "base64_encoded_enhanced_audio",
        "metadata": {
            "total_processing_time_ms": 1234,
            "download_time_ms": 123,        # Time to download from URL (if used)
            "inference_time_ms": 890,      # Time for AI processing
            "encoding_time_ms": 221,       # Time for base64 encoding
            "input_duration_s": 60.0,
            "rtf": 0.025,
            "model_used": "DeepFilterNet3",
            "gpu_used": "RTX 4090"
        }
    }
    """
    
    job_start_time = time.time()
    
    try:
        # Extract input data
        job_input = job.get('input', {})
        
        # Initialize timing variables
        download_time = 0.0
        decode_time = 0.0
        
        # Get audio data - either from Base64 or URL
        audio_url = job_input.get('audio_url')
        audio_base64 = job_input.get('audio_base64')
        
        if audio_url:
            # Download from URL
            logger.info(f"📥 Processing audio from URL: {audio_url}")
            try:
                audio_data, download_time = download_audio_from_url(audio_url)
            except ValueError as e:
                return handle_error(f"URL download failed: {str(e)}")
                
        elif audio_base64:
            # Decode from Base64
            logger.info("🔄 Processing audio from Base64 data")
            decode_start = time.time()
            try:
                audio_data = base64.b64decode(audio_base64)
                decode_time = time.time() - decode_start
                logger.info(f"✅ Decoded {len(audio_data)} bytes in {decode_time:.3f}s")
            except Exception as e:
                return handle_error(f"Base64 decode failed: {str(e)}")
        else:
            return handle_error("Missing required field: either 'audio_url' or 'audio_base64'")
        
        # Validate input format
        validation_error = validate_input(job_input)
        if validation_error:
            return handle_error(validation_error)
        
        # Load processor (cached after first call)
        proc = load_processor()
        
        # Process the audio
        logger.info("🎵 Starting audio enhancement...")
        inference_start = time.time()
        
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
            tmp_input.write(audio_data)
            tmp_input_path = tmp_input.name
        
        try:
            # Load and process audio
            enhanced_audio, metadata = proc.enhance_audio_file(
                input_path=tmp_input_path,
                sample_rate=job_input.get('sample_rate', 48000),
                attenuation_limit_db=job_input.get('attenuation_limit_db')
            )
            
            # Save enhanced audio to temporary file and encode
            encoding_start = time.time()
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                sf.write(tmp_output.name, enhanced_audio, metadata['sample_rate'])
                
                # Read back and encode as base64
                with open(tmp_output.name, 'rb') as f:
                    enhanced_audio_data = f.read()
                    enhanced_audio_base64 = base64.b64encode(enhanced_audio_data).decode('utf-8')
                
                # Clean up temp files
                os.unlink(tmp_output.name)
            
            encoding_time = time.time() - encoding_start
            inference_time = time.time() - inference_start
            
        finally:
            # Clean up input temp file
            os.unlink(tmp_input_path)
        
        total_time = time.time() - job_start_time
        
        # Prepare response
        response = {
            "enhanced_audio_base64": enhanced_audio_base64,
            "success": True
        }
        
        # Add metadata if requested
        if job_input.get('return_metadata', True):
            response["metadata"] = {
                **metadata,
                "total_processing_time_ms": round(total_time * 1000, 2),
                "download_time_ms": round(download_time * 1000, 2) if download_time > 0 else 0,
                "decode_time_ms": round(decode_time * 1000, 2) if decode_time > 0 else 0,
                "inference_time_ms": round(inference_time * 1000, 2),
                "encoding_time_ms": round(encoding_time * 1000, 2),
                "model_used": "DeepFilterNet3",
                "input_source": "URL" if audio_url else "Base64",
                "gpu_used": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "cuda_memory_used_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2) if torch.cuda.is_available() else 0
            }
        
        logger.info(f"✅ Audio enhanced successfully in {total_time:.2f}s (inference: {inference_time:.2f}s)")
        return response
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        logger.error(traceback.format_exc())
        return handle_error(f"Processing failed: {str(e)}")


def main():
    """Main function to start the serverless worker"""
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 Starting DeepFilterNet3 Serverless Worker")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"🔥 PyTorch version: {torch.__version__}")
    logger.info(f"🎮 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
        logger.info(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Warm up the processor (optional - improves first request performance)
    try:
        logger.info("🔥 Warming up processor...")
        warmup_start = time.time()
        load_processor()
        warmup_time = time.time() - warmup_start
        logger.info(f"🔥 Warmup completed in {warmup_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Warmup failed: {e}")
        sys.exit(1)
    
    # Start the RunPod serverless worker
    logger.info("🎯 Worker ready - waiting for requests...")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()