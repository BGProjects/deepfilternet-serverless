#!/usr/bin/env python3
"""
DeepFilterNet3 Serverless Handler for RunPod
Optimized for RTX 5090 GPU deployment with comprehensive error handling
"""

import os
import sys
import time
import base64
import tempfile
import requests
from typing import Dict, Any, Optional
import traceback
import platform
import psutil
import signal

import runpod
import torch
import numpy as np
import soundfile as sf
from loguru import logger

# Configure logging for RunPod visibility
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", level="INFO")
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", level="ERROR")

# Global error tracking
ERROR_COUNT = 0
MAX_ERRORS = 5

# Import PyTorch first to ensure CUDA libraries are loaded for ONNX Runtime
try:
    import onnxruntime
    if hasattr(onnxruntime, 'preload_dlls'):
        onnxruntime.preload_dlls()
    logger.info("ONNX Runtime CUDA libraries preloaded")
except Exception as e:
    logger.warning(f"Could not preload ONNX Runtime libraries: {e}")

# Import our custom modules
from audio_processor import DeepFilterNetProcessor
from utils import setup_logging, validate_input, handle_error

# Global processor instance (loaded once per container)
processor: Optional[DeepFilterNetProcessor] = None

def log_system_info():
    """Log comprehensive system information for debugging"""
    logger.info("🔍 SYSTEM INFORMATION COLLECTION STARTED")
    
    try:
        # Platform info
        logger.info(f"🖥️ Platform: {platform.platform()}")
        logger.info(f"🐍 Python: {platform.python_version()}")
        logger.info(f"🏗️ Architecture: {platform.architecture()}")
        logger.info(f"💾 Total RAM: {psutil.virtual_memory().total // (1024**3)}GB")
        logger.info(f"💾 Available RAM: {psutil.virtual_memory().available // (1024**3)}GB")
        logger.info(f"🔧 CPU Cores: {psutil.cpu_count()}")
        
        # CUDA/PyTorch info
        logger.info(f"🔥 PyTorch Version: {torch.__version__}")
        logger.info(f"🎮 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"🎮 CUDA Version: {torch.version.cuda}")
            logger.info(f"🎮 CUDA Device Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                gpu_capability = torch.cuda.get_device_capability(i)
                logger.info(f"🎮 GPU {i}: {gpu_name}")
                logger.info(f"🎮 GPU {i} Memory: {gpu_memory}GB")
                logger.info(f"🎮 GPU {i} Capability: sm_{gpu_capability[0]}{gpu_capability[1]}")
                
                # Check RTX 5090 specific capability
                if gpu_capability == (12, 0):  # sm_120
                    logger.warning(f"⚠️ RTX 5090 detected with sm_120 capability - potential compatibility issues")
        else:
            logger.warning("❌ CUDA not available - will use CPU fallback")
            
        # ONNX Runtime info
        try:
            import onnxruntime as ort
            logger.info(f"🧠 ONNX Runtime Version: {ort.__version__}")
            available_providers = ort.get_available_providers()
            logger.info(f"🧠 ONNX Providers: {available_providers}")
            
            if 'CUDAExecutionProvider' in available_providers:
                logger.info("✅ CUDA ExecutionProvider available")
            else:
                logger.warning("❌ CUDA ExecutionProvider not available")
                
        except Exception as e:
            logger.error(f"❌ ONNX Runtime info collection failed: {e}")
            
        # SciPy version check
        try:
            import scipy
            logger.info(f"📊 SciPy Version: {scipy.__version__}")
            # Check if RegularGridInterpolator is available
            from scipy.interpolate import RegularGridInterpolator
            logger.info("✅ RegularGridInterpolator available")
        except Exception as e:
            logger.error(f"❌ SciPy compatibility check failed: {e}")
            
        # Environment variables
        cuda_vars = [k for k in os.environ.keys() if 'CUDA' in k]
        for var in cuda_vars:
            logger.info(f"🔧 {var}: {os.environ[var]}")
            
        logger.info("✅ SYSTEM INFORMATION COLLECTION COMPLETED")
        
    except Exception as e:
        logger.error(f"❌ System info collection failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")

def setup_signal_handlers():
    """Setup signal handlers to catch segfaults and other crashes"""
    def signal_handler(signum, frame):
        logger.error(f"🚨 SIGNAL {signum} RECEIVED - WORKER CRASH DETECTED")
        logger.error(f"📋 Frame: {frame}")
        logger.error(f"📋 Stack trace: {traceback.format_stack()}")
        sys.exit(1)
    
    # Handle segfaults and other crash signals
    signal.signal(signal.SIGSEGV, signal_handler)  # Segmentation fault
    signal.signal(signal.SIGABRT, signal_handler)  # Abort
    signal.signal(signal.SIGFPE, signal_handler)   # Floating point exception
    signal.signal(signal.SIGILL, signal_handler)   # Illegal instruction

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
    RunPod serverless handler function with comprehensive error handling and logging
    
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
            "system_info": {...},
            "error_info": {...}  # If errors occurred
        }
    }
    """
    global ERROR_COUNT
    job_id = job.get('id', 'unknown')
    
    # Setup signal handlers
    setup_signal_handlers()
    
    logger.info(f"🚀 HANDLER STARTED - Job ID: {job_id}")
    job_start_time = time.time()
    
    try:
        # Input validation first
        job_input = job.get('input', {})
        validation_error = validate_input(job_input)
        
        if validation_error:
            logger.error(f"❌ Input validation failed: {validation_error}")
            return handle_error(validation_error, "VALIDATION_ERROR", job_id)
        
        # Log the input (without sensitive data)
        safe_input = {k: v for k, v in job_input.items() if 'base64' not in k.lower()}
        logger.info(f"📥 Job Input: {safe_input}")
        
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
                logger.error(f"❌ URL download failed: {e}")
                return handle_error(f"URL download failed: {str(e)}", "DOWNLOAD_ERROR", job_id)
                
        elif audio_base64:
            # Decode from Base64
            logger.info("🔄 Processing audio from Base64 data")
            decode_start = time.time()
            try:
                audio_data = base64.b64decode(audio_base64)
                decode_time = time.time() - decode_start
                logger.info(f"✅ Decoded {len(audio_data)} bytes in {decode_time:.3f}s")
            except Exception as e:
                logger.error(f"❌ Base64 decode failed: {e}")
                return handle_error(f"Base64 decode failed: {str(e)}", "DECODE_ERROR", job_id)
        else:
            error_msg = "Missing required field: either 'audio_url' or 'audio_base64'"
            logger.error(f"❌ {error_msg}")
            return handle_error(error_msg, "INPUT_MISSING", job_id)
        
        # Load processor with GPU/CPU fallback handling
        try:
            proc = load_processor()
        except Exception as e:
            logger.error(f"❌ Processor loading failed: {e}")
            # Try CPU fallback
            try:
                logger.info("🔄 Attempting CPU fallback...")
                proc = load_processor(use_cpu_fallback=True)
            except Exception as fallback_error:
                logger.error(f"❌ CPU fallback also failed: {fallback_error}")
                return handle_error(
                    f"Both GPU and CPU initialization failed: {e} | {fallback_error}", 
                    "PROCESSOR_INIT_ERROR", 
                    job_id
                )
        
        # Process the audio with comprehensive error handling
        logger.info("🎵 Starting audio enhancement...")
        inference_start = time.time()
        
        # Create temporary file for audio processing
        tmp_input_path = None
        tmp_output_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
                tmp_input.write(audio_data)
                tmp_input_path = tmp_input.name
            
            logger.info(f"📁 Created temp input file: {tmp_input_path}")
            
            # Load and process audio with detailed logging
            logger.info("🧠 Starting DeepFilterNet3 inference...")
            enhanced_audio, metadata = proc.enhance_audio_file(
                input_path=tmp_input_path,
                sample_rate=job_input.get('sample_rate', 48000),
                attenuation_limit_db=job_input.get('attenuation_limit_db'),
                chunk_duration_s=job_input.get('chunk_duration_s', 120)
            )
            
            logger.info(f"✅ Audio enhancement completed - Output duration: {len(enhanced_audio) / metadata['sample_rate']:.2f}s")
            
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
            
        except Exception as processing_error:
            logger.error(f"❌ Audio processing failed: {processing_error}")
            logger.error(f"📋 Processing error traceback: {traceback.format_exc()}")
            
            ERROR_COUNT += 1
            if ERROR_COUNT >= MAX_ERRORS:
                logger.error(f"🚨 MAX ERRORS ({MAX_ERRORS}) REACHED - WORKER RESTART RECOMMENDED")
            
            return handle_error(
                f"Audio processing failed: {processing_error}",
                "PROCESSING_ERROR",
                job_id
            )
            
        finally:
            # Clean up temp files safely
            try:
                if tmp_input_path and os.path.exists(tmp_input_path):
                    os.unlink(tmp_input_path)
                    logger.debug(f"🗑️ Cleaned up temp input file: {tmp_input_path}")
                if tmp_output_path and os.path.exists(tmp_output_path):
                    os.unlink(tmp_output_path)
                    logger.debug(f"🗑️ Cleaned up temp output file: {tmp_output_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Cleanup error: {cleanup_error}")
        
        total_time = time.time() - job_start_time
        
        # Prepare comprehensive response
        response = {
            "enhanced_audio_base64": enhanced_audio_base64,
            "success": True
        }
        
        # Add metadata if requested
        if job_input.get('return_metadata', True):
            # Get comprehensive system info
            system_info = {
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "cuda_memory_used_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2) if torch.cuda.is_available() else 0,
                "cuda_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 2) if torch.cuda.is_available() else 0,
            }
            
            response["metadata"] = {
                **metadata,
                "total_processing_time_ms": round(total_time * 1000, 2),
                "download_time_ms": round(download_time * 1000, 2) if download_time > 0 else 0,
                "decode_time_ms": round(decode_time * 1000, 2) if decode_time > 0 else 0,
                "inference_time_ms": round(inference_time * 1000, 2),
                "encoding_time_ms": round(encoding_time * 1000, 2),
                "model_used": "DeepFilterNet3",
                "input_source": "URL" if audio_url else "Base64",
                "system_info": system_info,
                "job_id": job_id,
                "error_count": ERROR_COUNT
            }
        
        logger.info(f"✅ Job {job_id} completed successfully in {total_time:.2f}s (inference: {inference_time:.2f}s)")
        
        # Reset error count on success
        ERROR_COUNT = 0
        
        # RunPod worker refresh mechanism (optional)
        # Check if worker refresh is requested by user or needed due to memory issues
        should_refresh = job_input.get('refresh_worker', False) or (torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory)
        
        if should_refresh:
            logger.info("🔄 Worker refresh requested - will restart after this job")
            response['refresh_worker'] = True
        
        return response
        
    except Exception as handler_error:
        logger.error(f"❌ Handler critical error: {handler_error}")
        logger.error(f"📋 Handler error traceback: {traceback.format_exc()}")
        
        ERROR_COUNT += 1
        
        # RunPod worker refresh on critical errors (prevent infinite crash loop)
        error_response = handle_error(
            f"Handler critical error: {handler_error}",
            "HANDLER_ERROR",
            job_id
        )
        
        # Force worker refresh if too many errors or segfault detected
        if ERROR_COUNT >= MAX_ERRORS or "SIGSEGV" in str(handler_error) or "segmentation fault" in str(handler_error).lower():
            logger.error(f"🚨 CRITICAL: Forcing worker refresh due to error count ({ERROR_COUNT}) or segfault")
            error_response['refresh_worker'] = True
        
        return error_response


def main():
    """Main function to start the serverless worker with comprehensive initialization"""
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 STARTING DEEPFILTERNET3 SERVERLESS WORKER")
    logger.info("=" * 60)
    
    # Setup signal handlers early
    setup_signal_handlers()
    
    # Log comprehensive system information
    log_system_info()
    
    logger.info("=" * 60)
    logger.info("🔥 PROCESSOR WARMUP STARTING...")
    
    # Warm up the processor (optional - improves first request performance)
    try:
        warmup_start = time.time()
        
        # Try GPU first, fallback to CPU if needed
        try:
            load_processor()
            logger.info("✅ GPU processor warmed up successfully")
        except Exception as gpu_error:
            logger.error(f"❌ GPU warmup failed: {gpu_error}")
            logger.info("🔄 Attempting CPU fallback warmup...")
            try:
                load_processor(use_cpu_fallback=True)
                logger.info("✅ CPU processor warmed up successfully")
            except Exception as cpu_error:
                logger.error(f"❌ CPU warmup also failed: {cpu_error}")
                logger.error("🚨 CRITICAL: Both GPU and CPU warmup failed")
                sys.exit(1)
        
        warmup_time = time.time() - warmup_start
        logger.info(f"🔥 Warmup completed in {warmup_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Warmup critical error: {e}")
        logger.error(f"📋 Warmup error traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🎯 WORKER READY - WAITING FOR REQUESTS...")
    logger.info("📊 Monitoring for exit code 139 (SIGSEGV) and other crashes...")
    logger.info("=" * 60)
    
    # Start the RunPod serverless worker
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.error(f"❌ RunPod worker startup failed: {e}")
        logger.error(f"📋 Startup error traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()