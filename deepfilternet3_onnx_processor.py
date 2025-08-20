#!/usr/bin/env python3
"""
DeepFilterNet3 ONNX Audio Enhancement Processor
Supports both regular and low-latency ONNX models
"""

import os
import sys
import warnings
import requests
import tempfile
import tarfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import configparser

import numpy as np
import onnxruntime as ort
import soundfile as sf
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
        self.min_nb_erb_freqs = self.config.getint('df', 'min_nb_erb_freqs', fallback=2)
        self.df_order = self.config.getint('df', 'df_order', fallback=5)
        self.df_lookahead = self.config.getint('df', 'df_lookahead', fallback=2)
        
        logger.info(f"Config loaded: SR={self.sr}, FFT={self.fft_size}, HOP={self.hop_size}")


class AudioFeatureExtractor:
    """Audio feature extraction for DeepFilterNet3"""
    
    def __init__(self, config: DeepFilterNet3Config):
        self.config = config
        self.window = self._create_window()
        
    def _create_window(self) -> np.ndarray:
        """Create Hann window for STFT"""
        return np.hanning(self.config.fft_size).astype(np.float32)
    
    def stft(self, audio: np.ndarray) -> np.ndarray:
        """Short-Time Fourier Transform"""
        from scipy.signal import stft as scipy_stft
        
        # Apply STFT with proper parameters
        f, t, spec = scipy_stft(
            audio,
            fs=self.config.sr,
            window=self.window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            nfft=self.config.fft_size,
            return_onesided=True
        )
        
        # Transpose to match expected format [T, F]
        return spec.T
    
    def erb_scale(self, frequencies: np.ndarray) -> np.ndarray:
        """Convert frequencies to ERB scale"""
        return 9.265 * np.log(1 + frequencies / (24.7 * 9.265))
    
    def create_erb_filterbank(self) -> np.ndarray:
        """Create ERB filterbank"""
        n_fft = self.config.fft_size
        n_freqs = n_fft // 2 + 1
        
        # Frequency bins
        freqs = np.linspace(0, self.config.sr / 2, n_freqs)
        
        # ERB boundaries
        erb_low = self.erb_scale(0)
        erb_high = self.erb_scale(self.config.sr / 2)
        erb_centers = np.linspace(erb_low, erb_high, self.config.nb_erb)
        
        # Create filterbank
        filterbank = np.zeros((self.config.nb_erb, n_freqs), dtype=np.float32)
        
        for i, erb_center in enumerate(erb_centers):
            # Simple triangular filters for ERB bands
            erb_freqs = self.erb_scale(freqs)
            
            if i == 0:
                left_edge = erb_low
            else:
                left_edge = erb_centers[i-1]
                
            if i == len(erb_centers) - 1:
                right_edge = erb_high
            else:
                right_edge = erb_centers[i+1]
            
            # Triangular filter
            mask = (erb_freqs >= left_edge) & (erb_freqs <= right_edge)
            if np.any(mask):
                # Linear interpolation for triangular shape
                left_slope = (erb_freqs - left_edge) / (erb_center - left_edge)
                right_slope = (right_edge - erb_freqs) / (right_edge - erb_center)
                
                filter_response = np.minimum(left_slope, right_slope)
                filter_response = np.maximum(0, filter_response)
                filterbank[i] = filter_response * mask
        
        return filterbank
    
    def extract_erb_features(self, spec: np.ndarray) -> np.ndarray:
        """Extract ERB features from spectrogram"""
        # Get magnitude spectrogram
        mag_spec = np.abs(spec)
        
        # Apply ERB filterbank
        erb_fb = self.create_erb_filterbank()
        erb_features = np.dot(mag_spec, erb_fb.T)
        
        # Convert to dB and normalize
        erb_features = 20 * np.log10(np.maximum(erb_features, 1e-10))
        
        # Normalize (simplified version)
        erb_features = (erb_features + 90) / 90  # Simple normalization
        
        return erb_features.astype(np.float32)
    
    def extract_spec_features(self, spec: np.ndarray, nb_df: int) -> np.ndarray:
        """Extract spectral features for DF decoder"""
        # Take only the first nb_df frequency bins
        spec_df = spec[:, :nb_df]
        
        # Convert to real/imaginary representation
        real_part = np.real(spec_df)
        imag_part = np.imag(spec_df)
        
        # Stack real and imaginary parts
        spec_features = np.stack([real_part, imag_part], axis=-1)
        
        # Unit normalization (simplified)
        norm = np.sqrt(np.sum(spec_features**2, axis=-1, keepdims=True))
        spec_features = spec_features / np.maximum(norm, 1e-10)
        
        return spec_features.astype(np.float32)


class DeepFilterNet3ONNX:
    """DeepFilterNet3 ONNX Processor"""
    
    def __init__(self, model_path: str, is_low_latency: bool = False):
        """
        Initialize DeepFilterNet3 ONNX processor
        
        Args:
            model_path: Path to extracted ONNX model directory or tar.gz file
            is_low_latency: Whether to use low-latency optimizations
        """
        self.is_low_latency = is_low_latency
        self.model_dir = self._extract_if_needed(model_path)
        
        # Load configuration
        config_path = os.path.join(self.model_dir, "config.ini")
        self.config = DeepFilterNet3Config(config_path)
        
        # Initialize feature extractor
        self.feature_extractor = AudioFeatureExtractor(self.config)
        
        # Load ONNX models
        self._load_onnx_models()
        
        logger.info(f"DeepFilterNet3 ONNX initialized (Low-latency: {is_low_latency})")
    
    def _extract_if_needed(self, model_path: str) -> str:
        """Extract tar.gz model if needed"""
        if model_path.endswith('.tar.gz'):
            # Extract to temporary directory
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
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        if self.is_low_latency:
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Load the three ONNX models
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
    
    def preprocess_audio(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess audio for ONNX inference"""
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add channel dimension
        
        # Ensure 48kHz sampling rate
        if audio.shape[0] > 1:
            # Convert to mono if stereo
            audio = np.mean(audio, axis=0, keepdims=True)
        
        features = {}
        
        for ch in range(audio.shape[0]):
            # Extract STFT
            spec = self.feature_extractor.stft(audio[ch])
            
            # Extract ERB features
            erb_feat = self.feature_extractor.extract_erb_features(spec)
            
            # Extract spectral features for DF
            spec_feat = self.feature_extractor.extract_spec_features(spec, self.config.nb_df)
            
            # Prepare features in required format
            # ERB features: [B, 1, T, E]
            erb_feat = erb_feat[np.newaxis, np.newaxis, :, :]
            
            # Spec features: [B, T, F, 2] -> [B, 2, T, F]
            spec_feat = spec_feat.transpose(2, 0, 1)[np.newaxis, :, :, :]
            
            features = {
                'feat_erb': erb_feat.astype(np.float32),
                'feat_spec': spec_feat.astype(np.float32),
                'original_spec': spec
            }
            
            break  # Process only first channel for now
        
        return features
    
    def run_inference(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run ONNX inference"""
        # 1. Encoder
        enc_inputs = {
            'feat_erb': features['feat_erb'],
            'feat_spec': features['feat_spec']
        }
        
        enc_outputs = self.enc_session.run(None, enc_inputs)
        e0, e1, e2, e3, emb, c0, lsnr = enc_outputs
        
        # 2. ERB Decoder
        erb_dec_inputs = {
            'emb': emb,
            'e3': e3,
            'e2': e2,
            'e1': e1,
            'e0': e0
        }
        
        erb_outputs = self.erb_dec_session.run(None, erb_dec_inputs)
        mask = erb_outputs[0]
        
        # 3. DF Decoder
        df_dec_inputs = {
            'emb': emb,
            'c0': c0
        }
        
        df_outputs = self.df_dec_session.run(None, df_dec_inputs)
        coefs = df_outputs[0]
        
        return {
            'mask': mask,
            'lsnr': lsnr,
            'coefs': coefs,
            'enhanced_features': {
                'e0': e0, 'e1': e1, 'e2': e2, 'e3': e3,
                'emb': emb, 'c0': c0
            }
        }
    
    def postprocess_audio(self, 
                         inference_results: Dict[str, np.ndarray], 
                         original_spec: np.ndarray) -> np.ndarray:
        """Postprocess inference results to audio"""
        mask = inference_results['mask']
        
        # Apply mask to original spectrogram
        # Simplified masking - in practice this would be more complex
        mask_squeezed = np.squeeze(mask)  # Remove batch and channel dims
        
        # Apply mask (assuming mask is for magnitude)
        enhanced_spec = original_spec * mask_squeezed
        
        # ISTFT to get time domain audio
        from scipy.signal import istft as scipy_istft
        
        _, enhanced_audio = scipy_istft(
            enhanced_spec.T,  # Transpose back to [F, T]
            fs=self.config.sr,
            window=self.feature_extractor.window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            nfft=self.config.fft_size
        )
        
        return enhanced_audio.astype(np.float32)
    
    def enhance_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Complete audio enhancement pipeline"""
        logger.info(f"Processing audio: shape={audio.shape}, dtype={audio.dtype}")
        
        # Preprocess
        features = self.preprocess_audio(audio)
        
        # Inference
        results = self.run_inference(features)
        
        # Postprocess
        enhanced_audio = self.postprocess_audio(results, features['original_spec'])
        
        # Metadata
        metadata = {
            'lsnr_db': float(np.mean(results['lsnr'])),
            'model_type': 'DeepFilterNet3_ll' if self.is_low_latency else 'DeepFilterNet3',
            'sample_rate': self.config.sr
        }
        
        logger.info(f"Enhancement complete. LSNR: {metadata['lsnr_db']:.2f} dB")
        
        return enhanced_audio, metadata


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
    # URLs and paths
    audio_url = "https://cozmo.com.tr/videotest.wav"
    models_dir = "/mnt/d/Yeni klasör (5)/DeepFilterNet-main/models"
    
    # Model paths
    dfn3_onnx_path = os.path.join(models_dir, "DeepFilterNet3_onnx.tar.gz")
    dfn3_ll_onnx_path = os.path.join(models_dir, "DeepFilterNet3_ll_onnx.tar.gz")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download audio file
        audio_file = os.path.join(temp_dir, "input_audio.wav")
        download_audio_file(audio_url, audio_file)
        
        # Load audio
        audio, original_sr = sf.read(audio_file)
        logger.info(f"Loaded audio: {audio.shape}, SR: {original_sr}")
        
        # Resample to 48kHz if needed
        if original_sr != 48000:
            import librosa
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=48000)
            logger.info("Audio resampled to 48kHz")
        
        # Process with both models
        models_to_test = [
            (dfn3_onnx_path, False, "DeepFilterNet3"),
            (dfn3_ll_onnx_path, True, "DeepFilterNet3_LowLatency")
        ]
        
        for model_path, is_ll, model_name in models_to_test:
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue
                
            logger.info(f"\n=== Processing with {model_name} ===")
            
            try:
                # Initialize processor
                processor = DeepFilterNet3ONNX(model_path, is_low_latency=is_ll)
                
                # Enhance audio
                enhanced_audio, metadata = processor.enhance_audio(audio)
                
                # Save enhanced audio
                output_file = os.path.join(temp_dir, f"enhanced_{model_name}.wav")
                sf.write(output_file, enhanced_audio, 48000)
                
                logger.info(f"Enhanced audio saved: {output_file}")
                logger.info(f"Metadata: {metadata}")
                
                # Copy to current directory for inspection
                import shutil
                final_output = f"enhanced_{model_name}.wav"
                shutil.copy2(output_file, final_output)
                logger.info(f"Final output: {final_output}")
                
            except Exception as e:
                logger.error(f"Error processing with {model_name}: {e}")
                import traceback
                traceback.print_exc()


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