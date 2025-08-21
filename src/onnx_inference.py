#!/usr/bin/env python3
"""
ONNX Inference Engine for DeepFilterNet3
Optimized for GPU acceleration and RTX 4090
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import onnxruntime as ort
from loguru import logger


class ONNXInferenceEngine:
    """ONNX Runtime inference engine for DeepFilterNet3"""
    
    def __init__(self, 
                 model_path: str, 
                 use_gpu: bool = True, 
                 optimize_for_rtx4090: bool = False):
        """
        Initialize ONNX inference engine
        
        Args:
            model_path: Path to directory containing ONNX models
            use_gpu: Enable GPU acceleration
            optimize_for_rtx4090: Apply RTX 4090 specific optimizations
        """
        
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu
        self.optimize_for_rtx4090 = optimize_for_rtx4090
        
        # Setup execution providers
        self.providers = self._setup_providers()
        
        # Setup session options
        self.session_options = self._setup_session_options()
        
        # Load ONNX models
        self.sessions = self._load_models()
        
        logger.info(f"🧠 ONNX inference engine ready")
        
        # Safe provider name extraction handling both string and tuple formats
        provider_names = []
        for p in self.providers:
            if isinstance(p, tuple):
                # Handle tuple format: ('CUDAExecutionProvider', {...})
                provider_names.append(p[0].replace('ExecutionProvider', ''))
            elif isinstance(p, str):
                # Handle string format: 'CUDAExecutionProvider'
                provider_names.append(p.replace('ExecutionProvider', ''))
            else:
                provider_names.append(str(p))
        
        logger.info(f"🎮 Providers: {provider_names}")
    
    def _setup_providers(self) -> List[str]:
        """Setup ONNX execution providers with priority order"""
        
        providers = []
        
        if self.use_gpu:
            # Check available providers
            available_providers = ort.get_available_providers()
            
            # RTX 4090 optimized provider configuration (disabled by default for stability)
            if 'TensorrtExecutionProvider' in available_providers and self.optimize_for_rtx4090 and os.getenv('ENABLE_TENSORRT', 'false').lower() == 'true':
                try:
                    providers.append(('TensorrtExecutionProvider', {
                        'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # Reduced to 4GB for stability
                        'trt_fp16_enable': True,
                        'trt_int8_enable': False,  # Keep quality
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '/tmp/trt_cache'
                    }))
                    logger.info("🚀 TensorRT provider enabled for RTX 4090")
                except Exception as e:
                    logger.warning(f"⚠️ TensorRT provider failed to initialize: {e}")
                    logger.info("🔄 Falling back to CUDA provider")
            
            if 'CUDAExecutionProvider' in available_providers:
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                
                # RTX 5090/4090 specific optimizations
                if self.optimize_for_rtx4090:
                    # Detect GPU type and set appropriate memory limits
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name()
                        if "RTX 5090" in gpu_name:
                            gpu_mem_limit = 28 * 1024 * 1024 * 1024  # 28GB for RTX 5090
                            logger.info("🎮 RTX 5090 detected - Using 28GB GPU memory limit")
                        elif "RTX 5080" in gpu_name:
                            gpu_mem_limit = 14 * 1024 * 1024 * 1024  # 14GB for RTX 5080  
                            logger.info("🎮 RTX 5080 detected - Using 14GB GPU memory limit")
                        else:
                            gpu_mem_limit = 20 * 1024 * 1024 * 1024  # 20GB for RTX 4090 and others
                            logger.info(f"🎮 {gpu_name} detected - Using 20GB GPU memory limit")
                    else:
                        gpu_mem_limit = 16 * 1024 * 1024 * 1024  # Default fallback
                    
                    cuda_options.update({
                        'gpu_mem_limit': gpu_mem_limit,
                        'arena_extend_strategy': 'kSameAsRequested',  # More aggressive for large VRAM
                        'cudnn_conv1d_pad_to_nc1d': True,  # Audio processing optimization
                        'enable_cuda_graph': False,  # Disabled for compatibility
                    })
                
                providers.append(('CUDAExecutionProvider', cuda_options))
                logger.info("🎮 CUDA provider enabled")
        
        # CPU fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def _setup_session_options(self) -> ort.SessionOptions:
        """Configure ONNX session options for optimal performance"""
        
        sess_options = ort.SessionOptions()
        
        # Execution mode
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Thread configuration
        if self.optimize_for_rtx4090:
            # RTX 4090 has excellent parallel processing capability
            sess_options.intra_op_num_threads = 8
            sess_options.inter_op_num_threads = 4
        else:
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2
        
        # Graph optimization
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Memory optimization for serverless
        if self.use_gpu:
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
        
        # Enable profiling for debugging (disable in production)
        # sess_options.enable_profiling = True
        
        return sess_options
    
    def _load_models(self) -> Dict[str, ort.InferenceSession]:
        """Load all ONNX model sessions"""
        
        model_files = {
            'encoder': 'enc.onnx',
            'erb_decoder': 'erb_dec.onnx',
            'df_decoder': 'df_dec.onnx'
        }
        
        sessions = {}
        total_load_time = 0
        
        for name, filename in model_files.items():
            model_file = self.model_path / filename
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            start_time = time.time()
            
            try:
                session = ort.InferenceSession(
                    str(model_file),
                    sess_options=self.session_options,
                    providers=self.providers
                )
                
                load_time = time.time() - start_time
                total_load_time += load_time
                
                sessions[name] = session
                
                # Log model info
                input_shapes = {inp.name: inp.shape for inp in session.get_inputs()}
                output_shapes = {out.name: out.shape for out in session.get_outputs()}
                
                logger.info(f"📂 Loaded {name}: {filename} ({load_time:.2f}s)")
                logger.debug(f"   Inputs: {input_shapes}")
                logger.debug(f"   Outputs: {output_shapes}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {name} ({filename}): {e}")
                raise
        
        logger.info(f"📚 All models loaded in {total_load_time:.2f}s total")
        return sessions
    
    def run_inference(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run complete DeepFilterNet3 inference pipeline
        
        Args:
            features: Dictionary with 'feat_erb' and 'feat_spec'
            
        Returns:
            Dictionary with inference results including 'mask', 'lsnr', 'coefs'
        """
        
        inference_start = time.time()
        
        try:
            # Stage 1: Encoder
            encoder_start = time.time()
            encoder_inputs = {
                'feat_erb': features['feat_erb'],
                'feat_spec': features['feat_spec']
            }
            
            encoder_outputs = self.sessions['encoder'].run(None, encoder_inputs)
            e0, e1, e2, e3, emb, c0, lsnr = encoder_outputs
            
            encoder_time = time.time() - encoder_start
            logger.debug(f"🧠 Encoder: {encoder_time*1000:.1f}ms")
            
            # Stage 2: ERB Decoder  
            erb_decoder_start = time.time()
            erb_decoder_inputs = {
                'emb': emb,
                'e3': e3,
                'e2': e2,
                'e1': e1,
                'e0': e0
            }
            
            erb_decoder_outputs = self.sessions['erb_decoder'].run(None, erb_decoder_inputs)
            mask = erb_decoder_outputs[0]
            
            erb_decoder_time = time.time() - erb_decoder_start
            logger.debug(f"🎭 ERB Decoder: {erb_decoder_time*1000:.1f}ms")
            
            # Stage 3: DF Decoder
            df_decoder_start = time.time()
            df_decoder_inputs = {
                'emb': emb,
                'c0': c0
            }
            
            df_decoder_outputs = self.sessions['df_decoder'].run(None, df_decoder_inputs)
            coefs = df_decoder_outputs[0]
            
            df_decoder_time = time.time() - df_decoder_start
            logger.debug(f"🔧 DF Decoder: {df_decoder_time*1000:.1f}ms")
            
            total_inference_time = time.time() - inference_start
            
            results = {
                'mask': mask,
                'lsnr': lsnr,
                'coefs': coefs,
                'intermediate_features': {
                    'e0': e0, 'e1': e1, 'e2': e2, 'e3': e3,
                    'emb': emb, 'c0': c0
                },
                'timing': {
                    'encoder_ms': round(encoder_time * 1000, 2),
                    'erb_decoder_ms': round(erb_decoder_time * 1000, 2),
                    'df_decoder_ms': round(df_decoder_time * 1000, 2),
                    'total_ms': round(total_inference_time * 1000, 2)
                }
            }
            
            logger.debug(f"⚡ Total inference: {total_inference_time*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ ONNX inference failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        info = {
            'providers': self.providers,
            'gpu_enabled': self.use_gpu,
            'rtx4090_optimized': self.optimize_for_rtx4090,
            'models': {}
        }
        
        for name, session in self.sessions.items():
            model_info = {
                'inputs': [
                    {
                        'name': inp.name,
                        'shape': inp.shape,
                        'type': inp.type
                    } for inp in session.get_inputs()
                ],
                'outputs': [
                    {
                        'name': out.name,
                        'shape': out.shape,
                        'type': out.type
                    } for out in session.get_outputs()
                ]
            }
            info['models'][name] = model_info
        
        return info
    
    def warm_up(self) -> Dict[str, float]:
        """
        Warm up the inference engine with dummy data
        Useful for reducing first-request latency in serverless environments
        """
        
        logger.info("🔥 Warming up ONNX inference engine...")
        
        # Create dummy inputs matching expected shapes
        dummy_inputs = {
            'feat_erb': np.random.randn(1, 1, 100, 32).astype(np.float32),
            'feat_spec': np.random.randn(1, 2, 100, 96).astype(np.float32)
        }
        
        warmup_start = time.time()
        
        try:
            # Run inference with dummy data
            results = self.run_inference(dummy_inputs)
            
            warmup_time = time.time() - warmup_start
            
            # Extract timing information
            timing_info = results.get('timing', {})
            timing_info['total_warmup_ms'] = round(warmup_time * 1000, 2)
            
            logger.info(f"🔥 Warmup completed in {warmup_time:.2f}s")
            
            return timing_info
            
        except Exception as e:
            logger.error(f"❌ Warmup failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        
        logger.info("🧹 Cleaning up ONNX sessions...")
        
        for name, session in self.sessions.items():
            try:
                # ONNX Runtime sessions are automatically cleaned up
                # but we can explicitly delete references
                del session
                logger.debug(f"🗑️ Cleaned up {name}")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up {name}: {e}")
        
        self.sessions.clear()
        logger.info("✅ Cleanup completed")