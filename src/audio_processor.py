#!/usr/bin/env python3
"""
DeepFilterNet3 Audio Processor
ONNX-based implementation optimized for GPU inference
"""

import os
import time
import configparser
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, resample
from loguru import logger

from onnx_inference import ONNXInferenceEngine


class AudioConfig:
    """Configuration management for DeepFilterNet3"""
    
    def __init__(self, config_path: str):
        """Load configuration from DeepFilterNet3 config.ini"""
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Audio processing parameters
        self.sr = self.config.getint('df', 'sr', fallback=48000)
        self.fft_size = self.config.getint('df', 'fft_size', fallback=960)
        self.hop_size = self.config.getint('df', 'hop_size', fallback=480)
        self.nb_erb = self.config.getint('df', 'nb_erb', fallback=32)
        self.nb_df = self.config.getint('df', 'nb_df', fallback=96)
        self.norm_tau = self.config.getfloat('df', 'norm_tau', fallback=1.0)
        self.lsnr_max = self.config.getfloat('df', 'lsnr_max', fallback=35.0)
        self.lsnr_min = self.config.getfloat('df', 'lsnr_min', fallback=-15.0)
        
        logger.info(f"📐 Config loaded: SR={self.sr}Hz, FFT={self.fft_size}, HOP={self.hop_size}")


class AudioFeatureExtractor:
    """Feature extraction for DeepFilterNet3 ONNX inference"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.window = np.hanning(config.fft_size).astype(np.float32)
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features required for ONNX inference
        
        Args:
            audio: Input audio signal [samples]
            
        Returns:
            Dictionary with 'feat_erb' and 'feat_spec' for ONNX input
        """
        
        # Compute STFT
        freqs, times, spec = stft(
            audio,
            fs=self.config.sr,
            window=self.window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            nfft=self.config.fft_size,
            return_onesided=True
        )
        
        # Transpose to [T, F] format for processing
        spec = spec.T
        
        # Extract ERB features
        erb_features = self._extract_erb_features(spec)
        
        # Extract spectral features for DF decoder
        spec_features = self._extract_spectral_features(spec)
        
        return {
            'feat_erb': erb_features,
            'feat_spec': spec_features,
            'original_spec': spec,
            'stft_params': {
                'freqs': freqs,
                'times': times,
                'window': self.window
            }
        }
    
    def _extract_erb_features(self, spec: np.ndarray) -> np.ndarray:
        """Extract ERB (Equivalent Rectangular Bandwidth) features"""
        
        # Get magnitude spectrogram
        mag_spec = np.abs(spec)
        
        # Simple ERB band grouping (frequency domain filtering)
        n_freqs = mag_spec.shape[1]
        n_erb = self.config.nb_erb
        
        # Create ERB bands by grouping frequencies
        erb_features = np.zeros((mag_spec.shape[0], n_erb), dtype=np.float32)
        
        for i in range(n_erb):
            start_idx = int(i * n_freqs / n_erb)
            end_idx = int((i + 1) * n_freqs / n_erb)
            
            if start_idx < end_idx:
                # Average magnitude in this ERB band
                erb_features[:, i] = np.mean(mag_spec[:, start_idx:end_idx], axis=1)
        
        # Convert to dB scale and normalize
        erb_features = 20 * np.log10(np.maximum(erb_features, 1e-10))
        erb_features = np.tanh(erb_features / 40.0)  # Normalize to [-1, 1]
        
        # Reshape for ONNX input: [B=1, C=1, T, E]
        erb_features = erb_features[np.newaxis, np.newaxis, :, :]
        
        return erb_features.astype(np.float32)
    
    def _extract_spectral_features(self, spec: np.ndarray) -> np.ndarray:
        """Extract spectral features for DF decoder"""
        
        # Take only the first nb_df frequency bins
        spec_df = spec[:, :self.config.nb_df]
        
        # Real and imaginary parts
        real_part = np.real(spec_df)
        imag_part = np.imag(spec_df)
        
        # Stack real/imaginary and normalize
        spec_features = np.stack([real_part, imag_part], axis=-1)
        
        # Unit normalization
        norm = np.sqrt(np.sum(spec_features**2, axis=-1, keepdims=True))
        spec_features = spec_features / np.maximum(norm, 1e-10)
        
        # Reshape for ONNX input: [B=1, C=2, T, F]
        spec_features = spec_features.transpose(2, 0, 1)[np.newaxis, :, :, :]
        
        return spec_features.astype(np.float32)


class DeepFilterNetProcessor:
    """Main DeepFilterNet3 audio processor using ONNX"""
    
    def __init__(self, model_path: str, use_gpu: bool = True, optimize_for_rtx4090: bool = False):
        """
        Initialize the DeepFilterNet processor
        
        Args:
            model_path: Path to directory containing ONNX models and config
            use_gpu: Enable GPU acceleration
            optimize_for_rtx4090: Apply RTX 4090 specific optimizations
        """
        
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu
        self.optimize_for_rtx4090 = optimize_for_rtx4090
        
        # Load configuration
        config_file = self.model_path / "config.ini"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        self.config = AudioConfig(str(config_file))
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor(self.config)
        self.inference_engine = ONNXInferenceEngine(
            model_path=str(self.model_path),
            use_gpu=use_gpu,
            optimize_for_rtx4090=optimize_for_rtx4090
        )
        
        logger.info(f"🎤 DeepFilterNet processor initialized")
        logger.info(f"🎮 GPU enabled: {use_gpu}")
        logger.info(f"⚡ RTX 4090 optimized: {optimize_for_rtx4090}")
    
    def enhance_audio_file(self, 
                          input_path: str, 
                          output_path: Optional[str] = None,
                          sample_rate: Optional[int] = None,
                          attenuation_limit_db: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhance audio from file
        
        Args:
            input_path: Input audio file path
            output_path: Optional output file path
            sample_rate: Optional target sample rate (default: 48kHz)
            attenuation_limit_db: Optional noise attenuation limit in dB
            
        Returns:
            Tuple of (enhanced_audio_array, metadata_dict)
        """
        
        start_time = time.time()
        
        # Load audio
        audio, orig_sr = sf.read(input_path)
        load_time = time.time() - start_time
        
        logger.info(f"🎵 Loaded audio: {audio.shape}, {orig_sr}Hz")
        
        # Process audio
        enhanced_audio, metadata = self.enhance_audio(
            audio=audio,
            orig_sample_rate=orig_sr,
            target_sample_rate=sample_rate or self.config.sr,
            attenuation_limit_db=attenuation_limit_db
        )
        
        # Add file metadata
        metadata.update({
            'load_time_ms': round(load_time * 1000, 2),
            'input_file': input_path,
            'input_duration_s': round(len(audio) / orig_sr, 2)
        })
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, enhanced_audio, metadata['sample_rate'])
            metadata['output_file'] = output_path
            logger.info(f"💾 Enhanced audio saved: {output_path}")
        
        return enhanced_audio, metadata
    
    def enhance_audio(self, 
                     audio: np.ndarray, 
                     orig_sample_rate: int,
                     target_sample_rate: int = 48000,
                     attenuation_limit_db: Optional[float] = None,
                     max_chunk_duration_s: float = 120.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhance audio array with automatic chunking for large files
        
        Args:
            audio: Input audio array
            orig_sample_rate: Original sample rate
            target_sample_rate: Target sample rate for processing
            attenuation_limit_db: Optional noise attenuation limit
            max_chunk_duration_s: Maximum chunk duration in seconds (default: 120s for RTX 5090)
            
        Returns:
            Tuple of (enhanced_audio, metadata)
        """
        
        process_start = time.time()
        
        # Calculate audio duration
        audio_duration_s = len(audio) / orig_sample_rate
        
        # If audio is too long, process in chunks
        if audio_duration_s > max_chunk_duration_s:
            logger.info(f"🔄 Audio duration {audio_duration_s:.1f}s > {max_chunk_duration_s}s, processing in chunks")
            return self._enhance_audio_chunked(audio, orig_sample_rate, target_sample_rate, 
                                             attenuation_limit_db, max_chunk_duration_s)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            logger.info("🔄 Converted stereo to mono")
        
        # Resample if needed
        if orig_sample_rate != target_sample_rate:
            resample_start = time.time()
            new_length = int(len(audio) * target_sample_rate / orig_sample_rate)
            audio = resample(audio, new_length)
            resample_time = time.time() - resample_start
            logger.info(f"🔄 Resampled: {orig_sample_rate}Hz → {target_sample_rate}Hz ({resample_time:.2f}s)")
        else:
            resample_time = 0
        
        # Extract features
        feature_start = time.time()
        features = self.feature_extractor.extract_features(audio)
        feature_time = time.time() - feature_start
        
        # Run ONNX inference
        inference_start = time.time()
        inference_results = self.inference_engine.run_inference(features)
        inference_time = time.time() - inference_start
        
        # Reconstruct enhanced audio
        reconstruct_start = time.time()
        enhanced_audio = self._reconstruct_audio(
            features, 
            inference_results, 
            attenuation_limit_db=attenuation_limit_db
        )
        reconstruct_time = time.time() - reconstruct_start
        
        total_time = time.time() - process_start
        
        # Prepare metadata
        metadata = {
            'sample_rate': target_sample_rate,
            'processing_time_s': round(total_time, 3),
            'rtf': round(total_time / (len(enhanced_audio) / target_sample_rate), 4),
            'feature_extraction_ms': round(feature_time * 1000, 2),
            'inference_time_ms': round(inference_time * 1000, 2),
            'reconstruction_ms': round(reconstruct_time * 1000, 2),
            'resample_time_ms': round(resample_time * 1000, 2),
            'lsnr_db': float(np.mean(inference_results['lsnr'])),
            'input_samples': len(audio),
            'output_samples': len(enhanced_audio)
        }
        
        logger.info(f"🎯 Enhancement complete: RTF={metadata['rtf']:.3f}, LSNR={metadata['lsnr_db']:.1f}dB")
        
        return enhanced_audio, metadata
    
    def _reconstruct_audio(self, 
                          features: Dict[str, np.ndarray], 
                          inference_results: Dict[str, np.ndarray],
                          attenuation_limit_db: Optional[float] = None) -> np.ndarray:
        """Reconstruct enhanced audio from inference results"""
        
        # Get the enhancement mask
        mask = inference_results['mask']
        original_spec = features['original_spec']
        stft_params = features['stft_params']
        
        # Apply mask to original spectrum
        enhanced_spec = self._apply_enhancement_mask(original_spec, mask, attenuation_limit_db)
        
        # Convert back to time domain using ISTFT
        _, enhanced_audio = istft(
            enhanced_spec.T,  # Transpose back to [F, T]
            fs=self.config.sr,
            window=stft_params['window'],
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            nfft=self.config.fft_size
        )
        
        return enhanced_audio.astype(np.float32)
    
    def _enhance_audio_chunked(self, 
                              audio: np.ndarray, 
                              orig_sample_rate: int,
                              target_sample_rate: int,
                              attenuation_limit_db: Optional[float],
                              chunk_duration_s: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process large audio files in chunks to avoid GPU memory overflow
        """
        
        process_start = time.time()
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            logger.info("🔄 Converted stereo to mono")
        
        # Resample if needed
        if orig_sample_rate != target_sample_rate:
            resample_start = time.time()
            new_length = int(len(audio) * target_sample_rate / orig_sample_rate)
            audio = resample(audio, new_length)
            resample_time = time.time() - resample_start
            logger.info(f"🔄 Resampled: {orig_sample_rate}Hz → {target_sample_rate}Hz ({resample_time:.2f}s)")
        else:
            resample_time = 0
        
        # Calculate chunk parameters
        samples_per_chunk = int(chunk_duration_s * target_sample_rate)
        overlap_samples = int(0.5 * target_sample_rate)  # 0.5s overlap
        num_chunks = max(1, int(np.ceil((len(audio) - overlap_samples) / (samples_per_chunk - overlap_samples))))
        
        logger.info(f"📊 Processing {len(audio)} samples in {num_chunks} chunks of {chunk_duration_s}s each")
        
        enhanced_chunks = []
        total_inference_time = 0
        total_feature_time = 0
        total_reconstruct_time = 0
        
        for i in range(num_chunks):
            chunk_start = i * (samples_per_chunk - overlap_samples)
            chunk_end = min(chunk_start + samples_per_chunk, len(audio))
            
            chunk_audio = audio[chunk_start:chunk_end]
            logger.info(f"🔄 Processing chunk {i+1}/{num_chunks}: {len(chunk_audio)} samples")
            
            # Extract features for this chunk
            feature_start = time.time()
            features = self.feature_extractor.extract_features(chunk_audio)
            feature_time = time.time() - feature_start
            total_feature_time += feature_time
            
            # Run ONNX inference
            inference_start = time.time()
            inference_results = self.inference_engine.run_inference(features)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Reconstruct enhanced audio
            reconstruct_start = time.time()
            enhanced_chunk = self._reconstruct_audio(
                features, 
                inference_results, 
                attenuation_limit_db=attenuation_limit_db
            )
            reconstruct_time = time.time() - reconstruct_start
            total_reconstruct_time += reconstruct_time
            
            # Handle overlap if not first chunk
            if i > 0 and len(enhanced_chunks) > 0:
                # Crossfade overlap region
                overlap_start = overlap_samples // 2
                overlap_end = overlap_samples
                
                # Apply crossfade to avoid discontinuities
                fade_in = np.linspace(0, 1, overlap_end - overlap_start)
                fade_out = np.linspace(1, 0, overlap_end - overlap_start)
                
                if len(enhanced_chunk) > overlap_end - overlap_start:
                    enhanced_chunk[overlap_start:overlap_end] = (
                        enhanced_chunk[overlap_start:overlap_end] * fade_in +
                        enhanced_chunks[-1][-len(fade_out):] * fade_out
                    )
                    # Remove overlapped part from previous chunk
                    enhanced_chunks[-1] = enhanced_chunks[-1][:-len(fade_out)]
            
            enhanced_chunks.append(enhanced_chunk)
        
        # Concatenate all chunks
        enhanced_audio = np.concatenate(enhanced_chunks)
        
        process_end = time.time()
        total_time = process_end - process_start
        
        # Prepare metadata
        metadata = {
            'sample_rate': target_sample_rate,
            'processing_time_s': round(total_time, 3),
            'rtf': round(total_time / (len(enhanced_audio) / target_sample_rate), 4),
            'feature_extraction_ms': round(total_feature_time * 1000, 2),
            'inference_time_ms': round(total_inference_time * 1000, 2),
            'reconstruction_ms': round(total_reconstruct_time * 1000, 2),
            'resample_time_ms': round(resample_time * 1000, 2),
            'input_samples': len(audio),
            'output_samples': len(enhanced_audio),
            'num_chunks': num_chunks,
            'chunk_duration_s': chunk_duration_s,
            'chunked_processing': True
        }
        
        logger.info(f"🎯 Chunked enhancement complete: {num_chunks} chunks, RTF={metadata['rtf']:.3f}")
        
        return enhanced_audio, metadata
    
    def _apply_enhancement_mask(self, 
                               spec: np.ndarray, 
                               mask: np.ndarray, 
                               attenuation_limit_db: Optional[float] = None) -> np.ndarray:
        """Apply the neural enhancement mask to the spectrum"""
        
        # Remove batch dimensions from mask
        mask = np.squeeze(mask)
        
        # Interpolate mask to match spectrum dimensions if needed
        if mask.shape != spec.shape:
            from scipy.interpolate import RegularGridInterpolator
            
            # Create interpolation function using RegularGridInterpolator
            x_orig = np.linspace(0, 1, mask.shape[1])
            y_orig = np.linspace(0, 1, mask.shape[0])
            
            x_new = np.linspace(0, 1, spec.shape[1])
            y_new = np.linspace(0, 1, spec.shape[0])
            
            # RegularGridInterpolator expects points as (y, x) for 2D data
            interp_func = RegularGridInterpolator((y_orig, x_orig), mask, method='linear', bounds_error=False, fill_value=0)
            
            # Create meshgrid for new coordinates
            y_mesh, x_mesh = np.meshgrid(y_new, x_new, indexing='ij')
            points = np.stack([y_mesh.ravel(), x_mesh.ravel()], axis=1)
            
            # Interpolate and reshape to target shape
            mask = interp_func(points).reshape(spec.shape)
        
        # Apply mask to spectrum
        enhanced_spec = spec * mask.astype(np.complex64)
        
        # Apply attenuation limit if specified
        if attenuation_limit_db is not None and abs(attenuation_limit_db) > 0:
            lim = 10 ** (-abs(attenuation_limit_db) / 20)
            enhanced_spec = spec * lim + enhanced_spec * (1 - lim)
        
        return enhanced_spec