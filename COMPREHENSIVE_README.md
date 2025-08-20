# DeepFilter: ECG Baseline Wander Removal - Comprehensive Guide

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Deep Learning Models](#deep-learning-models)
- [Classical Filters](#classical-filters)
- [Data Flow Explanation](#data-flow-explanation)
- [File Structure](#file-structure)
- [Scripts and Their Purposes](#scripts-and-their-purposes)
- [Complete Workflow](#complete-workflow)
- [Understanding the Results](#understanding-the-results)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## Overview

DeepFilter is a comprehensive ECG (Electrocardiogram) baseline wander removal system that compares deep learning approaches against classical digital filtering methods. The system trains 6 different deep learning models and compares them with 2 classical filters across multiple performance metrics.

**Key Features:**
- 6 Deep Learning Models (from simple to state-of-the-art)
- 2 Classical Digital Filters (FIR and IIR)
- Comprehensive performance evaluation
- ONNX model export for deployment
- Multiple visualization and analysis tools

## System Architecture

```
ECG Data Pipeline:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   QT Database   │    │  Add Baseline    │    │  Train/Test     │
│  (Clean ECG)    │ -> │  Wander Noise    │ -> │   Split         │
│                 │    │  (MIT-BIH NSTDB) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────────────────────────────────────────────────────┐
│                    Model Training & Testing                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Deep Learning  │  Classical      │         Evaluation          │
│     Models      │   Filters       │                             │
│   (6 models)    │  (FIR + IIR)    │  • Performance Metrics      │
│                 │                 │  • Visual Comparisons       │
│                 │                 │  • Statistical Analysis     │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Deep Learning Models

### 1. **DRNN (Deep Recurrent Neural Network)**
- **Architecture**: Recurrent neural network with LSTM/GRU layers
- **Purpose**: Captures temporal dependencies in ECG signals
- **Strengths**: Good at handling sequential data patterns
- **Use Case**: Baseline approach for time-series denoising

### 2. **FCN-DAE (Fully Convolutional Denoising Autoencoder)**
- **Architecture**: Encoder-decoder with convolutional layers
- **Purpose**: Learns compressed representations for denoising
- **Strengths**: Effective feature extraction and reconstruction
- **Use Case**: Standard autoencoder approach for signal denoising

### 3. **Vanilla L (Vanilla CNN Linear)**
- **Architecture**: Simple CNN with linear activations
- **Purpose**: Basic convolutional approach
- **Strengths**: Fast training and inference
- **Use Case**: Baseline CNN model for comparison

### 4. **Vanilla NL (Vanilla CNN Non-Linear)**
- **Architecture**: Simple CNN with non-linear activations (ReLU, etc.)
- **Purpose**: Enhanced version of Vanilla L
- **Strengths**: Better feature learning than linear version
- **Use Case**: Standard CNN approach with non-linearities

### 5. **Multibranch LANL (Multi-branch Linear and Non-Linear)**
- **Architecture**: Multiple parallel branches with different activation functions
- **Purpose**: Combines linear and non-linear feature extraction
- **Strengths**: Captures diverse signal characteristics
- **Use Case**: Advanced multi-path processing

### 6. **Multibranch LANLD (Multi-branch Linear and Non-Linear Dilated)** ⭐ **BEST MODEL**
- **Architecture**: Inception-inspired multi-branch with dilated convolutions
- **Purpose**: State-of-the-art approach with multi-scale feature extraction
- **Strengths**: 
  - Dilated convolutions capture different temporal scales
  - Multi-branch design processes signals at multiple resolutions
  - Best performance across all metrics
- **Use Case**: Production-ready model for ECG denoising

## Classical Filters

### 1. **FIR Filter (Finite Impulse Response)**
- **Type**: Linear phase digital filter
- **Characteristics**: Stable, linear phase response
- **Speed**: Slower processing (~45 seconds per signal)
- **Use Case**: High-quality filtering when speed is not critical

### 2. **IIR Filter (Infinite Impulse Response)**
- **Type**: Recursive digital filter
- **Characteristics**: Efficient, but potential stability issues
- **Speed**: Fast processing (~1 second per signal)
- **Use Case**: Real-time applications requiring fast filtering

## Data Flow Explanation

### 1. **Data Preparation Phase**
```python
# What happens in Data_Preparation()
QT_Database (clean ECG) + MIT_BIH_Noise (baseline wander) = Training Data
├── X_train: Noisy ECG signals (input)
├── y_train: Clean ECG signals (ground truth)
├── X_test: Noisy ECG test signals
└── y_test: Clean ECG test signals (ground truth)
```

### 2. **Training Phase**
```python
# For each model:
model.fit(X_train, y_train)  # Learn: noisy → clean mapping
model.save_weights(f'{model_name}_weights.best.hdf5')
```

### 3. **Testing Phase**
```python
# What's saved in test_results_*.pkl:
[X_test, y_test, y_pred] where:
├── X_test: Noisy input signals (shape: [samples, 512])
├── y_test: Ground truth clean signals (shape: [samples, 512])
└── y_pred: Model predictions (shape: [samples, 512])
```

### 4. **Noise Versions (nv1 and nv2)**
- **nv1**: First type of baseline wander noise characteristics
- **nv2**: Second type of baseline wander noise characteristics
- **Purpose**: Ensures robust evaluation across different noise conditions
- **Combination**: Results are concatenated for comprehensive analysis

## File Structure

```
DeepFilter/
├── 📁 Data_Preparation/          # Dataset creation and preprocessing
├── 📁 deepFilter/                # Deep learning models and training pipeline
├── 📁 digitalFilters/            # Classical FIR and IIR filter implementations
├── 📁 utils/                     # Metrics calculation and visualization tools
├── 📁 data/                      # ECG datasets (QT Database, MIT-BIH NSTDB)
├── 📁 onnx_models/              # Exported ONNX models for deployment
├── 📁 exported_models/          # TensorFlow SavedModel exports
├── 📁 final_results_*/          # Generated analysis results and plots
│
├── 🐍 DeepFilter_main.py        # Main training and testing script
├── 🐍 simple_export_all.py     # Export models to SavedModel format
├── 🐍 convert_to_onnx.py        # Convert SavedModel to ONNX format
├── 🐍 generate_final_plots.py   # Generate comprehensive analysis plots
├── 🐍 create_visual_comparison.py # ONNX model validation and comparison
├── 🐍 test_onnx_models_only.py  # Test ONNX model performance
├── 🐍 analyze_snr_performance.py # Signal-to-Noise Ratio analysis
│
├── 📊 *_weights.best.hdf5       # Trained model weights
├── 📊 test_results_*_nv*.pkl    # Test results for each model and noise version
├── 📊 timing_nv*.pkl            # Training and testing time measurements
└── 📊 *.png                     # Generated visualization plots
```

## Scripts and Their Purposes

### Core Training and Testing
- **`DeepFilter_main.py`**: Main script that trains all models and runs initial testing
- **`generate_final_plots.py`**: Generates comprehensive analysis from saved results

### Model Export and Deployment
- **`simple_export_all.py`**: Exports trained models to TensorFlow SavedModel format
- **`convert_to_onnx.py`**: Converts SavedModel to ONNX for cross-platform deployment
- **`test_onnx_models_only.py`**: Validates ONNX model performance

### Analysis and Visualization
- **`create_visual_comparison.py`**: Creates visual comparisons using ONNX models
- **`analyze_snr_performance.py`**: Performs Signal-to-Noise Ratio analysis

### Key Differences Between Scripts

| Script | Purpose | Data Source | Models Tested |
|--------|---------|-------------|---------------|
| `generate_final_plots.py` | Analyze original results | All `test_results_*.pkl` | All 8 methods |
| `create_visual_comparison.py` | Test ONNX deployment | One model's test data | ONNX + Classical |
| `test_onnx_models_only.py` | Validate ONNX conversion | Fresh inference | ONNX models only |

## Complete Workflow

### Step 1: Environment Setup
```bash
# Install Miniconda3
# Create environment
conda env create -f environment_gpu.yaml
conda activate DeepFilter-GPU
```

### Step 2: Data Download
```bash
# Windows
powershell -ExecutionPolicy Bypass -File '.\download_data.ps1'
# Linux/macOS
bash ./download_data.sh
```

### Step 3: Training (Main Experiment)
```bash
python DeepFilter_main.py
# This will:
# 1. Prepare datasets (nv1 and nv2)
# 2. Train all 6 deep learning models
# 3. Test all models + classical filters
# 4. Save results to test_results_*.pkl files
```

### Step 4: Generate Final Analysis
```bash
python generate_final_plots.py
# Creates comprehensive analysis with:
# - Performance tables
# - Metric comparisons
# - Signal visualizations
```

### Step 5: Model Export (Optional)
```bash
# Export to SavedModel format
python simple_export_all.py

# Convert to ONNX format
python convert_to_onnx.py

# Test ONNX models
python test_onnx_models_only.py
```

### Step 6: Additional Analysis (Optional)
```bash
# Create visual comparisons
python create_visual_comparison.py

# SNR analysis
python analyze_snr_performance.py
```

## Understanding the Results

### Performance Hierarchy (Typical Results)
1. **🥇 Multibranch LANLD** - Best overall performance
2. **🥈 Multibranch LANL** - Second best deep learning model
3. **🥉 FCN-DAE** - Good autoencoder performance
4. **Vanilla NL** - Decent CNN with non-linearities
5. **DRNN** - Good for temporal patterns
6. **Vanilla L** - Basic CNN baseline
7. **IIR Filter** - Fast classical method
8. **FIR Filter** - Slow but stable classical method

### Key Performance Improvements
- **Deep Learning vs Classical**: ~12x improvement in SSD metric
- **SNR Improvement**: 17.37 dB improvement (2.2x better than classical)
- **Speed**: ONNX models achieve real-time performance

## Performance Metrics

### 1. **SSD (Sum of Squared Differences)**
- **Formula**: `Σ(y_true - y_pred)²`
- **Interpretation**: Lower is better
- **Use**: Overall signal reconstruction quality

### 2. **MAD (Maximum Absolute Difference)**
- **Formula**: `max(|y_true - y_pred|)`
- **Interpretation**: Lower is better
- **Use**: Worst-case error measurement

### 3. **PRD (Percentage Root-mean-square Difference)**
- **Formula**: `100 * √(Σ(y_true - y_pred)²) / √(Σ(y_true)²)`
- **Interpretation**: Lower is better (percentage)
- **Use**: Normalized error measurement

### 4. **COS_SIM (Cosine Similarity)**
- **Formula**: `(y_true · y_pred) / (||y_true|| × ||y_pred||)`
- **Interpretation**: Higher is better (0-1 scale)
- **Use**: Shape similarity measurement

## Troubleshooting

### Common Issues and Solutions

#### 1. **Empty Plots or Tables**
```bash
# Check if results files exist
dir test_results_*.pkl

# Run the robust analysis script
python generate_final_plots.py
```

#### 2. **CUDA/GPU Issues**
```bash
# Use CPU environment instead
conda env create -f environment.yaml
conda activate DeepFilter
```

#### 3. **Memory Issues**
- Reduce batch size in training scripts
- Use CPU environment for lower memory usage
- Process fewer signals at once

#### 4. **Missing Dependencies**
```bash
# Reinstall environment
conda env remove -n DeepFilter-GPU
conda env create -f environment_gpu.yaml
```

#### 5. **Incomplete Results (Power Outage)**
```bash
# Use the robust final plots generator
python generate_final_plots.py
# This handles missing/corrupted files gracefully
```

### File Size Reference
- **Deep Learning Results**: ~136 MB each (test_results_*_nv*.pkl)
- **Classical Filter Results**: ~163 MB each (test_results_FIR/IIR_nv*.pkl)
- **Total Storage**: ~2.4 GB for complete results

### Expected Runtime
- **Full Training**: 8-12 hours (GPU) / 24-48 hours (CPU)
- **Analysis Generation**: 5-10 minutes
- **ONNX Export**: 2-5 minutes
- **Visual Comparisons**: 10-30 minutes (depending on FIR inclusion)

---

## Citation

When using DeepFilter, please cite the main repo:

```bibtex
@article{romero2021deepfilter,
    title={DeepFilter: an ECG baseline wander removal filter using deep learning techniques},
    author={Romero, Francisco P and Pi{\~n}ol, David C and V{\'a}zquez-Seisdedos, Carlos R},
    journal={Biomedical Signal Processing and Control},
    volume={70},
    pages={102992},
    year={2021},
    publisher={Elsevier}
}
```

## License

MIT License - See LICENSE file for details.

---

*This comprehensive guide covers the complete DeepFilter system. For specific technical details, refer to the individual script documentation and the original research paper.*
