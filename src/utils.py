#!/usr/bin/env python3
"""
Utility functions for DeepFilterNet3 Serverless
"""

import sys
import base64
from typing import Dict, Any, Optional
import logging

from loguru import logger


def setup_logging(level: str = "INFO", format_string: Optional[str] = None):
    """Setup logging configuration for serverless environment"""
    
    # Remove default loguru handler
    logger.remove()
    
    # Custom format for serverless
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add handler for stdout (RunPod captures this)
    logger.add(
        sys.stdout,
        level=level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Suppress some noisy third-party loggers
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger.info(f"📋 Logging setup complete (level: {level})")


def validate_input(job_input: Dict[str, Any]) -> Optional[str]:
    """
    Validate RunPod job input
    
    Args:
        job_input: Input dictionary from RunPod job
        
    Returns:
        Error message if validation fails, None if valid
    """
    
    # Check required fields - must have either audio_base64 or audio_url
    audio_base64 = job_input.get('audio_base64')
    audio_url = job_input.get('audio_url')
    
    if not audio_base64 and not audio_url:
        return "Must provide either 'audio_base64' or 'audio_url'"
    
    if audio_base64 and audio_url:
        return "Cannot provide both 'audio_base64' and 'audio_url' - choose one"
    
    # Validate base64 encoding if provided
    if audio_base64:
        if not isinstance(audio_base64, str):
            return "Field 'audio_base64' must be a base64 encoded string"
        
        try:
            audio_data = base64.b64decode(audio_base64)
            if len(audio_data) == 0:
                return "Empty audio data"
            if len(audio_data) > 500 * 1024 * 1024:  # 500MB limit
                return "Audio data too large (max 500MB)"
        except Exception as e:
            return f"Invalid base64 audio data: {str(e)}"
    
    # Validate URL if provided
    if audio_url:
        if not isinstance(audio_url, str):
            return "Field 'audio_url' must be a string"
        
        if not (audio_url.startswith('http://') or audio_url.startswith('https://')):
            return "audio_url must be a valid HTTP/HTTPS URL"
    
    # Validate optional parameters
    sample_rate = job_input.get('sample_rate')
    if sample_rate is not None:
        if not isinstance(sample_rate, int) or sample_rate < 8000 or sample_rate > 96000:
            return "sample_rate must be integer between 8000 and 96000"
    
    output_format = job_input.get('output_format', 'wav')
    if output_format not in ['wav', 'flac', 'mp3']:
        return "output_format must be one of: wav, flac, mp3"
    
    attenuation_limit_db = job_input.get('attenuation_limit_db')
    if attenuation_limit_db is not None:
        if not isinstance(attenuation_limit_db, (int, float)):
            return "attenuation_limit_db must be a number"
        if abs(attenuation_limit_db) > 60:
            return "attenuation_limit_db must be between -60 and 60 dB"
    
    return None  # Valid input


def handle_error(error_message: str) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_message: Error description
        
    Returns:
        Error response dictionary
    """
    
    logger.error(f"❌ {error_message}")
    
    return {
        "error": error_message,
        "success": False,
        "enhanced_audio_base64": None,
        "metadata": {
            "error": True,
            "error_message": error_message
        }
    }


def calculate_rtf(processing_time_s: float, audio_duration_s: float) -> float:
    """
    Calculate Real-Time Factor (RTF)
    
    Args:
        processing_time_s: Time taken to process in seconds
        audio_duration_s: Duration of audio in seconds
        
    Returns:
        RTF value (lower is better, <1.0 means faster than real-time)
    """
    
    if audio_duration_s <= 0:
        return float('inf')
    
    return processing_time_s / audio_duration_s


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_audio_info(audio_data: bytes) -> Dict[str, Any]:
    """
    Extract basic information from audio data
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        Dictionary with audio information
    """
    
    import tempfile
    import soundfile as sf
    
    info = {
        'size_bytes': len(audio_data),
        'size_formatted': format_file_size(len(audio_data))
    }
    
    try:
        # Write to temp file to analyze
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file.flush()
            
            # Get audio properties
            audio, sample_rate = sf.read(tmp_file.name)
            
            info.update({
                'sample_rate': sample_rate,
                'channels': 1 if audio.ndim == 1 else audio.shape[1],
                'samples': len(audio) if audio.ndim == 1 else audio.shape[0],
                'duration_s': len(audio) / sample_rate if audio.ndim == 1 else audio.shape[0] / sample_rate,
                'format_detected': True
            })
            
    except Exception as e:
        logger.warning(f"Could not analyze audio format: {e}")
        info['format_detected'] = False
        info['error'] = str(e)
    
    return info


def create_test_input(duration_s: float = 5.0, sample_rate: int = 48000) -> Dict[str, Any]:
    """
    Create test input for local development
    
    Args:
        duration_s: Duration of test audio in seconds
        sample_rate: Sample rate of test audio
        
    Returns:
        Test input dictionary compatible with handler
    """
    
    import numpy as np
    import tempfile
    import soundfile as sf
    
    # Generate test audio (sine wave + noise)
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))
    
    # Create a mix of sine waves with noise
    clean_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 220 * t)    # A3 note
    )
    
    # Add noise
    noise = np.random.normal(0, 0.1, len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # Normalize
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal)) * 0.8
    
    # Save to temp file and encode
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
        sf.write(tmp_file.name, noisy_signal, sample_rate)
        
        with open(tmp_file.name, 'rb') as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    return {
        "input": {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "return_metadata": True
        }
    }


def log_system_info():
    """Log system information for debugging"""
    
    import platform
    import psutil
    import torch
    
    logger.info("🖥️  System Information:")
    logger.info(f"   Platform: {platform.platform()}")
    logger.info(f"   Python: {platform.python_version()}")
    logger.info(f"   CPU: {platform.processor()}")
    logger.info(f"   CPU cores: {psutil.cpu_count()}")
    logger.info(f"   Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    logger.info("🔥 PyTorch Information:")
    logger.info(f"   Version: {torch.__version__}")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"   CUDA version: {torch.version.cuda}")
        logger.info(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    try:
        import onnxruntime as ort
        logger.info("🧠 ONNX Runtime Information:")
        logger.info(f"   Version: {ort.__version__}")
        logger.info(f"   Available providers: {ort.get_available_providers()}")
    except ImportError:
        logger.warning("⚠️  ONNX Runtime not available")


class PerformanceMonitor:
    """Simple performance monitoring for serverless functions"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing a operation"""
        import time
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        import time
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        self.metrics[name] = duration
        del self.start_times[name]
        return duration
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics"""
        return self.metrics.copy()
    
    def log_metrics(self):
        """Log all metrics"""
        logger.info("📊 Performance Metrics:")
        for name, duration in self.metrics.items():
            logger.info(f"   {name}: {duration*1000:.2f}ms")


# Global performance monitor instance
perf_monitor = PerformanceMonitor()