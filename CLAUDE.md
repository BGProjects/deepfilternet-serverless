# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeepFilterNet is a speech enhancement framework for noise suppression using deep filtering. It supports full-band audio (48kHz) and provides real-time noise reduction capabilities through multiple model architectures (DeepFilterNet, DeepFilterNet2, DeepFilterNet3).

## Development Commands

### Rust Components
- **Build all Rust components**: `cargo build --release`
- **Build specific component**: `cargo build --release -p <package>` (libDF, pyDF, pyDF-data, ladspa, demo)
- **Run demo**: `cargo +nightly run -p df-demo --features ui --bin df-demo --release`
- **Run tests**: `cargo test`

### Python Components
- **Install Python dependencies**: `poetry -C DeepFilterNet install -E train -E eval`
- **Build Python extensions**: 
  - `maturin develop --release -m pyDF/Cargo.toml`
  - `maturin develop --release -m pyDF-data/Cargo.toml`
- **PyTorch installation shortcuts**:
  - CUDA 11: `poetry -C DeepFilterNet run poe install-torch-cuda11`
  - CUDA 12: `poetry -C DeepFilterNet run poe install-torch-cuda12`
  - CPU: `poetry -C DeepFilterNet run poe install-torch-cpu`

### Model Training
- **Train model**: `python DeepFilterNet/df/train.py <dataset_config> <data_dir> <base_dir>`
- **Prepare datasets**: `python DeepFilterNet/df/scripts/prepare_data.py --sr 48000 <type> <audio_files> <output.hdf5>`

### Audio Enhancement
- **Enhance audio (Rust)**: `cargo run --release --bin deep-filter -- <audio_file.wav>`
- **Enhance audio (Python)**: `python DeepFilterNet/df/enhance.py -m DeepFilterNet2 <audio_file.wav>`
- **CLI shortcut**: `deepFilter <audio_file.wav>` (after pip installation)

## Architecture Overview

### Multi-Language Architecture
- **Rust Core (`libDF`)**: High-performance audio processing, STFT/ISTFT, data loading, and augmentation
- **Python Framework (`DeepFilterNet`)**: Neural network models, training, evaluation, and visualization
- **Python Bindings**: `pyDF` (audio processing) and `pyDF-data` (dataset loading)

### Key Components
- **`libDF/src/lib.rs`**: Core Rust library with DFState, STFT processing, and ERB frequency conversion
- **`DeepFilterNet/df/train.py`**: Main training script with PyTorch implementation
- **`DeepFilterNet/df/enhance.py`**: Audio enhancement pipeline with model loading
- **`DeepFilterNet/df/model.py`**: Neural network architectures (DeepFilterNet, DeepFilterNet2, DeepFilterNet3)
- **`ladspa/`**: Real-time LADSPA plugin for PipeWire/PulseAudio integration
- **`demo/`**: Real-time demonstration application with UI

### Workspace Structure
The repository uses a Cargo workspace with these members:
- `libDF`: Core Rust audio processing library
- `pyDF`: Python bindings for audio processing
- `pyDF-data`: Python bindings for dataset functionality
- `ladspa`: LADSPA plugin for real-time use
- `demo`: Interactive demonstration application

### Dataset Format
- Uses HDF5 datasets for training data
- Separate datasets for speech, noise, and room impulse responses (RIRs)
- Dataset configuration in JSON format specifying train/valid/test splits
- Audio preprocessing at 48kHz sampling rate

## Code Standards
- **Rust formatting**: Uses standard `rustfmt` configuration
- **Python formatting**: Black with 100 character line length, isort for imports
- **Python style**: Target versions 3.8-3.10
- **Dependencies**: Poetry for Python dependency management, Cargo for Rust

## Model Files
Pre-trained models are stored in `models/` directory as compressed archives:
- DeepFilterNet.zip, DeepFilterNet2.zip, DeepFilterNet3.zip
- ONNX versions available for deployment
- Models automatically downloaded when using Python API