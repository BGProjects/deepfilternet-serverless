#!/usr/bin/env python3
"""
Simplified DeepFilterNet3 ONNX Processor (without librosa dependency)
"""

import os
import sys
import warnings
import requests
import tempfile
import tarfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import configparser

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import stft as scipy_stft, istft as scipy_istft
from loguru import logger

# Suppress warnings
warnings.filterwarnings("ignore")


class DeepFilterNet3Config:
    """Configuration parser for DeepFilterNet3"""
    
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Audio parameters from config
        self.sr = self.config.getint('df', 'sr', fallback=48000)
        self.fft_size = self.config.getint('df', 'fft_size', fallback=960)
        self.hop_size = self.config.getint('df', 'hop_size', fallback=480)
        self.nb_erb = self.config.getint('df', 'nb_erb', fallback=32)
        self.nb_df = self.config.getint('df', 'nb_df', fallback=96)
        self.norm_tau = self.config.getfloat('df', 'norm_tau', fallback=1.0)
        self.lsnr_max = self.config.getfloat('df', 'lsnr_max', fallback=35.0)
        self.lsnr_min = self.config.getfloat('df', 'lsnr_min', fallback=-15.0)
        
        logger.info(f"Config loaded: SR={self.sr}, FFT={self.fft_size}, HOP={self.hop_size}")


def simple_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling using scipy"""
    from scipy.signal import resample
    
    if orig_sr == target_sr:
        return audio
    
    # Calculate new length
    new_length = int(len(audio) * target_sr / orig_sr)
    
    # Resample
    resampled = resample(audio, new_length)
    
    logger.info(f"Resampled from {orig_sr}Hz to {target_sr}Hz: {len(audio)} -> {len(resampled)} samples")
    return resampled.astype(np.float32)


class SimpleFeatureExtractor:
    """Simplified audio feature extraction"""
    
    def __init__(self, config: DeepFilterNet3Config):
        self.config = config
        self.window = np.hanning(self.config.fft_size).astype(np.float32)
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features for ONNX model"""
        
        # Compute STFT
        f, t, spec = scipy_stft(
            audio,
            fs=self.config.sr,
            window=self.window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            nfft=self.config.fft_size,
            return_onesided=True
        )
        
        # Transpose to [T, F] format
        spec = spec.T
        
        # Extract ERB features (simplified)
        erb_features = self._extract_erb_simple(spec)
        
        # Extract spectral features for DF decoder
        spec_features = self._extract_spec_simple(spec)
        
        return {
            'feat_erb': erb_features,
            'feat_spec': spec_features,
            'original_spec': spec
        }
    
    def _extract_erb_simple(self, spec: np.ndarray) -> np.ndarray:
        """Simplified ERB feature extraction"""
        
        # Get magnitude spectrum
        mag_spec = np.abs(spec)
        
        # Create simple ERB bands by frequency grouping
        n_freqs = mag_spec.shape[1]
        n_erb = self.config.nb_erb
        
        # Group frequencies into ERB bands
        erb_features = np.zeros((mag_spec.shape[0], n_erb), dtype=np.float32)
        
        for i in range(n_erb):
            start_freq = int(i * n_freqs / n_erb)
            end_freq = int((i + 1) * n_freqs / n_erb)
            
            if start_freq < end_freq:
                # Average magnitude in this band
                erb_features[:, i] = np.mean(mag_spec[:, start_freq:end_freq], axis=1)
        
        # Convert to dB and normalize
        erb_features = 20 * np.log10(np.maximum(erb_features, 1e-10))
        erb_features = np.tanh(erb_features / 40.0)  # Simple normalization
        
        # Reshape for ONNX: [B, 1, T, E]
        erb_features = erb_features[np.newaxis, np.newaxis, :, :]
        
        return erb_features.astype(np.float32)
    
    def _extract_spec_simple(self, spec: np.ndarray) -> np.ndarray:
        """Simplified spectral feature extraction"""
        
        # Take only the first nb_df frequency bins
        spec_df = spec[:, :self.config.nb_df]
        
        # Real and imaginary parts
        real_part = np.real(spec_df)
        imag_part = np.imag(spec_df)
        
        # Stack and normalize
        spec_features = np.stack([real_part, imag_part], axis=-1)
        
        # Simple normalization
        norm = np.sqrt(np.sum(spec_features**2, axis=-1, keepdims=True))
        spec_features = spec_features / np.maximum(norm, 1e-10)
        
        # Reshape for ONNX: [B, 2, T, F]
        spec_features = spec_features.transpose(2, 0, 1)[np.newaxis, :, :, :]
        
        return spec_features.astype(np.float32)


class SimpleDeepFilterNet3ONNX:
    """Simplified DeepFilterNet3 ONNX Processor"""
    
    def __init__(self, model_path: str, is_low_latency: bool = False):
        self.is_low_latency = is_low_latency
        self.model_dir = self._extract_if_needed(model_path)
        
        # Load configuration
        config_path = os.path.join(self.model_dir, "config.ini")
        self.config = DeepFilterNet3Config(config_path)
        
        # Initialize feature extractor
        self.feature_extractor = SimpleFeatureExtractor(self.config)
        
        # Load ONNX models
        self._load_onnx_models()
        
        logger.info(f"DeepFilterNet3 ONNX initialized (Low-latency: {is_low_latency})")
    
    def _extract_if_needed(self, model_path: str) -> str:
        """Extract tar.gz model if needed"""
        if model_path.endswith('.tar.gz'):
            extract_dir = tempfile.mkdtemp()
            with tarfile.open(model_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            # Find the export directory
            for root, dirs, files in os.walk(extract_dir):
                if 'config.ini' in files:
                    return root
            
            raise FileNotFoundError("Could not find config.ini in extracted archive")
        else:
            return model_path
    
    def _load_onnx_models(self):
        """Load ONNX model sessions"""
        providers = ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        if self.is_low_latency:
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
        
        # Load models
        self.enc_session = ort.InferenceSession(
            os.path.join(self.model_dir, "enc.onnx"),
            sess_options,
            providers=providers
        )
        
        self.erb_dec_session = ort.InferenceSession(
            os.path.join(self.model_dir, "erb_dec.onnx"),
            sess_options,
            providers=providers
        )
        
        self.df_dec_session = ort.InferenceSession(
            os.path.join(self.model_dir, "df_dec.onnx"),
            sess_options,
            providers=providers
        )
        
        logger.info("ONNX models loaded successfully")
    
    def enhance_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Complete audio enhancement pipeline"""
        logger.info(f"Processing audio: shape={audio.shape}, dtype={audio.dtype}")
        
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Run ONNX inference
        try:
            # 1. Encoder
            enc_outputs = self.enc_session.run(None, {
                'feat_erb': features['feat_erb'],
                'feat_spec': features['feat_spec']
            })
            
            e0, e1, e2, e3, emb, c0, lsnr = enc_outputs
            logger.info(f"Encoder completed. LSNR shape: {lsnr.shape}")
            
            # 2. ERB Decoder
            erb_outputs = self.erb_dec_session.run(None, {
                'emb': emb,
                'e3': e3,
                'e2': e2,
                'e1': e1,
                'e0': e0
            })
            
            mask = erb_outputs[0]
            logger.info(f"ERB Decoder completed. Mask shape: {mask.shape}")
            
            # 3. DF Decoder
            df_outputs = self.df_dec_session.run(None, {
                'emb': emb,
                'c0': c0
            })
            
            coefs = df_outputs[0]
            logger.info(f"DF Decoder completed. Coefs shape: {coefs.shape}")
            
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            raise
        
        # Apply mask to original spectrum (simplified)
        enhanced_spec = self._apply_mask(features['original_spec'], mask)
        
        # Convert back to time domain
        enhanced_audio = self._spec_to_audio(enhanced_spec)
        
        # Metadata
        metadata = {
            'lsnr_db': float(np.mean(lsnr)),
            'model_type': 'DeepFilterNet3_ll' if self.is_low_latency else 'DeepFilterNet3',
            'sample_rate': self.config.sr
        }
        
        logger.info(f"Enhancement complete. LSNR: {metadata['lsnr_db']:.2f} dB")
        
        return enhanced_audio, metadata
    
    def _apply_mask(self, spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply enhancement mask to spectrum"""
        
        # Remove batch and channel dimensions from mask
        mask_squeezed = np.squeeze(mask)
        
        # Ensure mask has correct shape
        if mask_squeezed.shape != spec.shape:
            # Simple interpolation if shapes don't match
            from scipy.interpolate import interp1d
            
            # Interpolate mask to match spectrum size
            mask_interp = np.zeros_like(spec, dtype=np.float32)
            
            for t in range(spec.shape[0]):
                if t < mask_squeezed.shape[0]:
                    # Interpolate frequency dimension
                    f_orig = np.linspace(0, 1, mask_squeezed.shape[1])
                    f_new = np.linspace(0, 1, spec.shape[1])
                    interp_func = interp1d(f_orig, mask_squeezed[t], kind='linear', fill_value='extrapolate')
                    mask_interp[t] = interp_func(f_new)
                else:
                    mask_interp[t] = mask_interp[t-1]  # Repeat last frame
            
            mask_squeezed = mask_interp
        
        # Apply mask (magnitude scaling)
        enhanced_spec = spec * mask_squeezed.astype(np.complex64)
        
        return enhanced_spec
    
    def _spec_to_audio(self, spec: np.ndarray) -> np.ndarray:
        """Convert spectrum back to time domain audio"""
        
        _, enhanced_audio = scipy_istft(
            spec.T,  # Transpose back to [F, T]
            fs=self.config.sr,
            window=self.feature_extractor.window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            nfft=self.config.fft_size
        )
        
        return enhanced_audio.astype(np.float32)


def download_audio_file(url: str, target_path: str) -> str:
    """Download audio file from URL"""
    logger.info(f"Downloading audio from: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Audio downloaded to: {target_path}")
    return target_path


def main():
    """Main processing function"""
    audio_url = "https://cozmo.com.tr/videotest.wav"
    models_dir = "/mnt/d/Yeni klasör (5)/DeepFilterNet-main/models"
    
    # Model paths
    dfn3_onnx_path = os.path.join(models_dir, "DeepFilterNet3_onnx.tar.gz")
    dfn3_ll_onnx_path = os.path.join(models_dir, "DeepFilterNet3_ll_onnx.tar.gz")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download audio
        audio_file = os.path.join(temp_dir, "input_audio.wav")
        download_audio_file(audio_url, audio_file)
        
        # Load audio
        audio, original_sr = sf.read(audio_file)
        logger.info(f"Loaded audio: {audio.shape}, SR: {original_sr}")
        
        # Resample to 48kHz if needed
        if original_sr != 48000:
            audio = simple_resample(audio, original_sr, 48000)
        
        # Test normal model
        if os.path.exists(dfn3_onnx_path):
            logger.info("\n=== Testing DeepFilterNet3 Normal ===")
            
            try:
                processor = SimpleDeepFilterNet3ONNX(dfn3_onnx_path, is_low_latency=False)
                enhanced_audio, metadata = processor.enhance_audio(audio)
                
                output_file = "enhanced_dfn3_normal.wav"
                sf.write(output_file, enhanced_audio, 48000)
                
                logger.info(f"✓ Normal model completed: {output_file}")
                logger.info(f"  Metadata: {metadata}")
                
            except Exception as e:
                logger.error(f"✗ Normal model failed: {e}")
        
        # Test low-latency model  
        if os.path.exists(dfn3_ll_onnx_path):
            logger.info("\n=== Testing DeepFilterNet3 Low-Latency ===")
            
            try:
                processor_ll = SimpleDeepFilterNet3ONNX(dfn3_ll_onnx_path, is_low_latency=True)
                enhanced_audio_ll, metadata_ll = processor_ll.enhance_audio(audio)
                
                output_file_ll = "enhanced_dfn3_ll.wav"
                sf.write(output_file_ll, enhanced_audio_ll, 48000)
                
                logger.info(f"✓ Low-latency model completed: {output_file_ll}")
                logger.info(f"  Metadata: {metadata_ll}")
                
            except Exception as e:
                logger.error(f"✗ Low-latency model failed: {e}")


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)