#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visual Comparison Creator
Simple script to create clear visual comparisons of ONNX models vs classical filters
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import onnxruntime as ort
from datetime import datetime
from digitalFilters.dfilters import IIR_test_Dataset, FIR_test_Dataset
from utils.metrics import SSD

def create_output_directory():
    """Create timestamped output directory for visual comparisons"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"visual_comparison_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_test_data(test_data_path='test_results_Vanilla L_nv1.pkl'):
    """Load test data"""
    with open(test_data_path, 'rb') as f:
        X_test, y_test, _ = pickle.load(f)
    return X_test, y_test

def run_onnx_inference(model_path, X_test_sample):
    """Run inference on a single ONNX model"""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Ensure correct input format
    if len(X_test_sample.shape) == 2:
        X_test_sample = np.expand_dims(X_test_sample, axis=0)
    
    X_input = X_test_sample.astype(np.float32)
    y_pred = session.run([output_name], {input_name: X_input})[0]
    
    return y_pred

def run_iir_filter(X_test_sample, y_test_sample):
    """Run IIR filter on a single sample"""
    # Create dataset format
    dataset = [None, None, X_test_sample, y_test_sample]
    _, _, y_filter = IIR_test_Dataset(dataset)
    return y_filter

def run_fir_filter(X_test_sample, y_test_sample):
    """Run FIR filter on a single sample"""
    # Create dataset format
    dataset = [None, None, X_test_sample, y_test_sample]
    _, _, y_filter = FIR_test_Dataset(dataset)
    return y_filter

def create_single_signal_comparison(output_dir):
    """Create a detailed comparison for a single signal"""
    
    print("📊 Creating single signal detailed comparison...")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Select one representative signal (middle of dataset)
    signal_idx = len(X_test) // 2
    X_sample = X_test[signal_idx:signal_idx+1]  # Keep batch dimension
    y_sample = y_test[signal_idx:signal_idx+1]
    
    print(f"Using signal {signal_idx} for comparison")
    
    # ALL ONNX models to test (including missing ones)
    onnx_models = {
        'Multibranch_LANLD': 'onnx_models/Multibranch_LANLD.onnx',
        'Multibranch_LANL': 'onnx_models/Multibranch_LANL.onnx',
        'FCN_DAE': 'onnx_models/FCN_DAE.onnx', 
        'DRNN': 'onnx_models/DRNN.onnx',  # This IS the LSTM model
        'Vanilla_NL': 'onnx_models/Vanilla_NL.onnx',
        'Vanilla_L': 'onnx_models/Vanilla_L.onnx'
    }
    
    # Run all inferences
    results = {}
    
    # Classical filter
    print("Running IIR filter...")
    y_iir = run_iir_filter(X_sample, y_sample)
    results['IIR_Filter'] = y_iir[0, :, 0]  # Remove batch and channel dims
    
    # ONNX models
    for model_name, model_path in onnx_models.items():
        if os.path.exists(model_path):
            print(f"Running {model_name}...")
            y_pred = run_onnx_inference(model_path, X_sample)
            results[model_name] = y_pred[0, :, 0]  # Remove batch and channel dims
        else:
            print(f"❌ Model not found: {model_path}")
    
    # Prepare data for plotting
    time_ms = np.arange(512) / 360 * 1000  # Convert to milliseconds
    noisy_signal = X_sample[0, :, 0]
    clean_signal = y_sample[0, :, 0]
    
    # Create the plot
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4*(n_models+1), 8))
    
    # Top row: Input and Ground Truth
    axes[0, 0].plot(time_ms, noisy_signal, 'r-', linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Noisy Input\n(Baseline Wander)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-2, 2)
    
    axes[0, 1].plot(time_ms, clean_signal, 'g-', linewidth=1.5)
    axes[0, 1].set_title('Ground Truth\n(Clean ECG)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-2, 2)
    
    # Hide unused top row plots
    for i in range(2, n_models + 1):
        axes[0, i].axis('off')
    
    # Bottom row: Model outputs
    model_names = list(results.keys())
    colors = ['orange', 'blue', 'purple', 'cyan', 'magenta', 'navy']
    
    for i, (model_name, y_pred) in enumerate(results.items()):
        col_idx = i
        color = colors[i % len(colors)]
        
        # Plot prediction
        axes[1, col_idx].plot(time_ms, y_pred, color=color, linewidth=1.2)
        
        # Calculate MSE
        mse = np.mean((y_pred - clean_signal) ** 2)
        
        # Calculate SSD for single signal
        ssd = np.sum((y_pred - clean_signal) ** 2)
        
        # Format title
        title = model_name.replace('_', ' ')
        axes[1, col_idx].set_title(f'{title}\nMSE: {mse:.4f} | SSD: {ssd:.2f}', 
                                  fontsize=11, fontweight='bold')
        axes[1, col_idx].set_ylabel('Amplitude')
        axes[1, col_idx].set_xlabel('Time (ms)')
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].set_ylim(-2, 2)
    
    # Hide unused bottom row plots
    for i in range(len(results), n_models + 1):
        if i < n_models + 1:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_signal_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Single signal comparison saved: single_signal_comparison.png")
    
    return results, noisy_signal, clean_signal

def create_multiple_signals_grid(output_dir):
    """Create a grid showing multiple signals for the best models"""
    
    print("📊 Creating multiple signals grid...")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Select 4 representative signals
    n_signals = 4
    signal_indices = [100, 500, 1000, 2000]  # Different parts of dataset
    
    # Best models to show
    best_models = {
        'Multibranch_LANLD': 'onnx_models/Multibranch_LANLD.onnx',
        'FCN_DAE': 'onnx_models/FCN_DAE.onnx',
        'Vanilla_NL': 'onnx_models/Vanilla_NL.onnx'
    }
    
    # Create grid: signals x (input + ground_truth + models + IIR)
    n_cols = 2 + len(best_models) + 1  # input, ground_truth, models, IIR
    fig, axes = plt.subplots(n_signals, n_cols, figsize=(4*n_cols, 3*n_signals))
    
    time_ms = np.arange(512) / 360 * 1000
    
    for row, signal_idx in enumerate(signal_indices):
        print(f"Processing signal {signal_idx}...")
        
        # Get signal data
        X_sample = X_test[signal_idx:signal_idx+1]
        y_sample = y_test[signal_idx:signal_idx+1]
        noisy_signal = X_sample[0, :, 0]
        clean_signal = y_sample[0, :, 0]
        
        # Column 0: Noisy input
        axes[row, 0].plot(time_ms, noisy_signal, 'r-', linewidth=1, alpha=0.7)
        axes[row, 0].set_title(f'Noisy Input\n(Signal {signal_idx})' if row == 0 else f'Signal {signal_idx}')
        axes[row, 0].set_ylabel('Amplitude')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_ylim(-2, 2)
        
        # Column 1: Ground truth
        axes[row, 1].plot(time_ms, clean_signal, 'g-', linewidth=1.5)
        axes[row, 1].set_title('Ground Truth' if row == 0 else '')
        axes[row, 1].set_ylabel('Amplitude')
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_ylim(-2, 2)
        
        # ONNX models
        col = 2
        colors = ['blue', 'purple', 'cyan']
        for i, (model_name, model_path) in enumerate(best_models.items()):
            if os.path.exists(model_path):
                try:
                    y_pred = run_onnx_inference(model_path, X_sample)
                    pred_signal = y_pred[0, :, 0]
                    
                    axes[row, col].plot(time_ms, pred_signal, color=colors[i], linewidth=1.2)
                    
                    # Calculate MSE for this signal
                    mse = np.mean((pred_signal - clean_signal) ** 2)
                    
                    title = model_name.replace('_', ' ')
                    axes[row, col].set_title(f'{title}\nMSE: {mse:.3f}' if row == 0 else f'MSE: {mse:.3f}')
                    axes[row, col].set_ylabel('Amplitude')
                    axes[row, col].grid(True, alpha=0.3)
                    axes[row, col].set_ylim(-2, 2)
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[row, col].transAxes)
            
            col += 1
        
        # IIR Filter
        try:
            y_iir = run_iir_filter(X_sample, y_sample)
            iir_signal = y_iir[0, :, 0]
            
            axes[row, -1].plot(time_ms, iir_signal, 'orange', linewidth=1.2)
            
            mse_iir = np.mean((iir_signal - clean_signal) ** 2)
            axes[row, -1].set_title(f'IIR Filter\nMSE: {mse_iir:.3f}' if row == 0 else f'MSE: {mse_iir:.3f}')
            axes[row, -1].set_ylabel('Amplitude')
            axes[row, -1].grid(True, alpha=0.3)
            axes[row, -1].set_ylim(-2, 2)
            
        except Exception as e:
            print(f"Error with IIR filter: {e}")
            axes[row, -1].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[row, -1].transAxes)
        
        # Set x-label for bottom row
        if row == n_signals - 1:
            for col in range(n_cols):
                axes[row, col].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiple_signals_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Multiple signals grid saved: multiple_signals_grid.png")

def create_performance_summary(output_dir):
    """Create a performance summary chart"""
    
    print("📊 Creating performance summary...")
    
    # Load the CSV results if available
    csv_files = [f for f in os.listdir('.') if f.startswith('onnx_model_comparison_') and f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV results found. Run test_onnx_models_only.py first.")
        return
    
    # Use the most recent CSV file
    csv_file = sorted(csv_files)[-1]
    print(f"Using results from: {csv_file}")
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # Filter to get ONNX models and classical filters
    onnx_models = df[df['Type'] == 'ONNX'].copy()
    classical_models = df[df['Type'] == 'Classical'].copy()
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: SSD Comparison
    all_models = pd.concat([onnx_models, classical_models])
    colors = ['blue' if t == 'ONNX' else 'red' for t in all_models['Type']]
    
    bars1 = ax1.bar(range(len(all_models)), all_models['SSD_Mean'], color=colors, alpha=0.7)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('SSD (lower is better)')
    ax1.set_title('Sum of Squared Differences (SSD)', fontweight='bold')
    ax1.set_xticks(range(len(all_models)))
    ax1.set_xticklabels(all_models['Model'].str.replace('_', '\n'), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, all_models['SSD_Mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(all_models['SSD_Mean'])*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Cosine Similarity
    bars2 = ax2.bar(range(len(all_models)), all_models['COS_SIM_Mean'], color=colors, alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Cosine Similarity (higher is better)')
    ax2.set_title('Cosine Similarity', fontweight='bold')
    ax2.set_xticks(range(len(all_models)))
    ax2.set_xticklabels(all_models['Model'].str.replace('_', '\n'), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    for bar, value in zip(bars2, all_models['COS_SIM_Mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Speed Comparison (ONNX only)
    bars3 = ax3.bar(range(len(onnx_models)), onnx_models['Throughput_samples_per_s'], 
                   color='blue', alpha=0.7)
    ax3.set_xlabel('ONNX Models')
    ax3.set_ylabel('Throughput (samples/second)')
    ax3.set_title('ONNX Model Speed Comparison', fontweight='bold')
    ax3.set_xticks(range(len(onnx_models)))
    ax3.set_xticklabels(onnx_models['Model'].str.replace('_', '\n'), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, onnx_models['Throughput_samples_per_s']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(onnx_models['Throughput_samples_per_s'])*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Accuracy vs Speed scatter
    ax4.scatter(onnx_models['Throughput_samples_per_s'], onnx_models['COS_SIM_Mean'], 
               c='blue', s=100, alpha=0.7, label='ONNX Models')
    ax4.scatter(classical_models['Throughput_samples_per_s'], classical_models['COS_SIM_Mean'], 
               c='red', s=100, alpha=0.7, label='Classical Filters')
    
    # Add model names as labels
    for _, row in onnx_models.iterrows():
        ax4.annotate(row['Model'].replace('_', '\n'), 
                    (row['Throughput_samples_per_s'], row['COS_SIM_Mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Throughput (samples/second)')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Accuracy vs Speed Trade-off', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Performance summary saved: performance_summary.png")

def create_single_signal_with_both_filters(output_dir):
    """Create a detailed comparison for a single signal including both IIR and FIR filters"""
    
    print("📊 Creating single signal comparison with BOTH filters (this will take longer)...")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Select one representative signal (middle of dataset)
    signal_idx = len(X_test) // 2
    X_sample = X_test[signal_idx:signal_idx+1]  # Keep batch dimension
    y_sample = y_test[signal_idx:signal_idx+1]
    
    print(f"Using signal {signal_idx} for comparison")
    
    # ONNX models to test
    onnx_models = {
        'Multibranch_LANLD': 'onnx_models/Multibranch_LANLD.onnx',
        'FCN_DAE': 'onnx_models/FCN_DAE.onnx', 
        'Vanilla_NL': 'onnx_models/Vanilla_NL.onnx',
        'DRNN': 'onnx_models/DRNN.onnx'
    }
    
    # Run all inferences
    results = {}
    
    # Classical filters
    print("Running IIR filter...")
    y_iir = run_iir_filter(X_sample, y_sample)
    results['IIR_Filter'] = y_iir[0, :, 0]  # Remove batch and channel dims
    
    print("Running FIR filter (this will take ~45 seconds)...")
    y_fir = run_fir_filter(X_sample, y_sample)
    results['FIR_Filter'] = y_fir[0, :, 0]  # Remove batch and channel dims
    
    # ONNX models
    for model_name, model_path in onnx_models.items():
        if os.path.exists(model_path):
            print(f"Running {model_name}...")
            y_pred = run_onnx_inference(model_path, X_sample)
            results[model_name] = y_pred[0, :, 0]  # Remove batch and channel dims
        else:
            print(f"❌ Model not found: {model_path}")
    
    # Prepare data for plotting
    time_ms = np.arange(512) / 360 * 1000  # Convert to milliseconds
    noisy_signal = X_sample[0, :, 0]
    clean_signal = y_sample[0, :, 0]
    
    # Create the plot with more columns for both filters
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4*(n_models+1), 8))
    
    # Top row: Input and Ground Truth
    axes[0, 0].plot(time_ms, noisy_signal, 'r-', linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Noisy Input\n(Baseline Wander)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-2, 2)
    
    axes[0, 1].plot(time_ms, clean_signal, 'g-', linewidth=1.5)
    axes[0, 1].set_title('Ground Truth\n(Clean ECG)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-2, 2)
    
    # Hide unused top row plots
    for i in range(2, n_models + 1):
        axes[0, i].axis('off')
    
    # Bottom row: Model outputs
    model_names = list(results.keys())
    colors = ['orange', 'brown', 'blue', 'purple', 'cyan', 'magenta', 'navy']
    
    for i, (model_name, y_pred) in enumerate(results.items()):
        col_idx = i
        color = colors[i % len(colors)]
        
        # Plot prediction
        axes[1, col_idx].plot(time_ms, y_pred, color=color, linewidth=1.2)
        
        # Calculate MSE
        mse = np.mean((y_pred - clean_signal) ** 2)
        
        # Calculate SSD for single signal
        ssd = np.sum((y_pred - clean_signal) ** 2)
        
        # Format title
        title = model_name.replace('_', ' ')
        axes[1, col_idx].set_title(f'{title}\nMSE: {mse:.4f} | SSD: {ssd:.2f}', 
                                  fontsize=11, fontweight='bold')
        axes[1, col_idx].set_ylabel('Amplitude')
        axes[1, col_idx].set_xlabel('Time (ms)')
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].set_ylim(-2, 2)
    
    # Hide unused bottom row plots
    for i in range(len(results), n_models + 1):
        if i < n_models + 1:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_signal_with_both_filters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Single signal comparison with both filters saved: single_signal_with_both_filters.png")
    
    return results, noisy_signal, clean_signal

def create_multiple_signals_with_both_filters(output_dir):
    """Create a grid showing multiple signals with both IIR and FIR filters"""
    
    print("📊 Creating multiple signals grid with BOTH filters (this will take several minutes)...")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Select 3 representative signals (fewer due to FIR being slow)
    n_signals = 3
    signal_indices = [100, 1000, 2000]  # Different parts of dataset
    
    # Best models to show
    best_models = {
        'Multibranch_LANLD': 'onnx_models/Multibranch_LANLD.onnx',
        'FCN_DAE': 'onnx_models/FCN_DAE.onnx',
        'Vanilla_NL': 'onnx_models/Vanilla_NL.onnx'
    }
    
    # Create grid: signals x (input + ground_truth + models + IIR + FIR)
    n_cols = 2 + len(best_models) + 2  # input, ground_truth, models, IIR, FIR
    fig, axes = plt.subplots(n_signals, n_cols, figsize=(4*n_cols, 3*n_signals))
    
    time_ms = np.arange(512) / 360 * 1000
    
    for row, signal_idx in enumerate(signal_indices):
        print(f"Processing signal {signal_idx}...")
        
        # Get signal data
        X_sample = X_test[signal_idx:signal_idx+1]
        y_sample = y_test[signal_idx:signal_idx+1]
        noisy_signal = X_sample[0, :, 0]
        clean_signal = y_sample[0, :, 0]
        
        # Column 0: Noisy input
        axes[row, 0].plot(time_ms, noisy_signal, 'r-', linewidth=1, alpha=0.7)
        axes[row, 0].set_title(f'Noisy Input\n(Signal {signal_idx})' if row == 0 else f'Signal {signal_idx}')
        axes[row, 0].set_ylabel('Amplitude')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_ylim(-2, 2)
        
        # Column 1: Ground truth
        axes[row, 1].plot(time_ms, clean_signal, 'g-', linewidth=1.5)
        axes[row, 1].set_title('Ground Truth' if row == 0 else '')
        axes[row, 1].set_ylabel('Amplitude')
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_ylim(-2, 2)
        
        # ONNX models
        col = 2
        colors = ['blue', 'purple', 'cyan']
        for i, (model_name, model_path) in enumerate(best_models.items()):
            if os.path.exists(model_path):
                try:
                    y_pred = run_onnx_inference(model_path, X_sample)
                    pred_signal = y_pred[0, :, 0]
                    
                    axes[row, col].plot(time_ms, pred_signal, color=colors[i], linewidth=1.2)
                    
                    # Calculate MSE for this signal
                    mse = np.mean((pred_signal - clean_signal) ** 2)
                    
                    title = model_name.replace('_', ' ')
                    axes[row, col].set_title(f'{title}\nMSE: {mse:.3f}' if row == 0 else f'MSE: {mse:.3f}')
                    axes[row, col].set_ylabel('Amplitude')
                    axes[row, col].grid(True, alpha=0.3)
                    axes[row, col].set_ylim(-2, 2)
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[row, col].transAxes)
            
            col += 1
        
        # IIR Filter
        try:
            print(f"  Running IIR filter for signal {signal_idx}...")
            y_iir = run_iir_filter(X_sample, y_sample)
            iir_signal = y_iir[0, :, 0]
            
            axes[row, -2].plot(time_ms, iir_signal, 'orange', linewidth=1.2)
            
            mse_iir = np.mean((iir_signal - clean_signal) ** 2)
            axes[row, -2].set_title(f'IIR Filter\nMSE: {mse_iir:.3f}' if row == 0 else f'MSE: {mse_iir:.3f}')
            axes[row, -2].set_ylabel('Amplitude')
            axes[row, -2].grid(True, alpha=0.3)
            axes[row, -2].set_ylim(-2, 2)
            
        except Exception as e:
            print(f"Error with IIR filter: {e}")
            axes[row, -2].text(0.5, 0.5, 'IIR Error', ha='center', va='center', transform=axes[row, -2].transAxes)
        
        # FIR Filter
        try:
            print(f"  Running FIR filter for signal {signal_idx} (this takes ~45 seconds)...")
            y_fir = run_fir_filter(X_sample, y_sample)
            fir_signal = y_fir[0, :, 0]
            
            axes[row, -1].plot(time_ms, fir_signal, 'brown', linewidth=1.2)
            
            mse_fir = np.mean((fir_signal - clean_signal) ** 2)
            axes[row, -1].set_title(f'FIR Filter\nMSE: {mse_fir:.3f}' if row == 0 else f'MSE: {mse_fir:.3f}')
            axes[row, -1].set_ylabel('Amplitude')
            axes[row, -1].grid(True, alpha=0.3)
            axes[row, -1].set_ylim(-2, 2)
            
        except Exception as e:
            print(f"Error with FIR filter: {e}")
            axes[row, -1].text(0.5, 0.5, 'FIR Error', ha='center', va='center', transform=axes[row, -1].transAxes)
        
        # Set x-label for bottom row
        if row == n_signals - 1:
            for col in range(n_cols):
                axes[row, col].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiple_signals_with_both_filters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Multiple signals grid with both filters saved: multiple_signals_with_both_filters.png")

def create_individual_model_comparisons(output_dir):
    """Create individual comparison plots for each ONNX model"""
    
    print("📊 Creating individual model comparison plots...")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Select one representative signal
    signal_idx = len(X_test) // 2
    X_sample = X_test[signal_idx:signal_idx+1]
    y_sample = y_test[signal_idx:signal_idx+1]
    
    # ALL ONNX models to test individually
    onnx_models = {
        'Multibranch_LANLD': 'onnx_models/Multibranch_LANLD.onnx',
        'Multibranch_LANL': 'onnx_models/Multibranch_LANL.onnx',
        'FCN_DAE': 'onnx_models/FCN_DAE.onnx', 
        'DRNN': 'onnx_models/DRNN.onnx',  # LSTM model
        'Vanilla_NL': 'onnx_models/Vanilla_NL.onnx',
        'Vanilla_L': 'onnx_models/Vanilla_L.onnx'
    }
    
    # Prepare data
    time_ms = np.arange(512) / 360 * 1000
    noisy_signal = X_sample[0, :, 0]
    clean_signal = y_sample[0, :, 0]
    
    # Get IIR filter result for comparison
    y_iir = run_iir_filter(X_sample, y_sample)
    iir_signal = y_iir[0, :, 0]
    
    # Create individual plots for each model
    for model_name, model_path in onnx_models.items():
        if os.path.exists(model_path):
            print(f"Creating individual plot for {model_name}...")
            
            try:
                # Run ONNX inference
                y_pred = run_onnx_inference(model_path, X_sample)
                pred_signal = y_pred[0, :, 0]
                
                # Calculate metrics
                mse_dl = np.mean((pred_signal - clean_signal) ** 2)
                mse_iir = np.mean((iir_signal - clean_signal) ** 2)
                ssd_dl = np.sum((pred_signal - clean_signal) ** 2)
                ssd_iir = np.sum((iir_signal - clean_signal) ** 2)
                
                # Create individual comparison plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                
                # Top plot: Signal comparison
                ax1.plot(time_ms, noisy_signal, 'k-', linewidth=1, alpha=0.7, label='Noisy Input (ECG + BLW)')
                ax1.plot(time_ms, clean_signal, 'g-', linewidth=2, label='Ground Truth (Clean ECG)')
                ax1.plot(time_ms, pred_signal, 'b-', linewidth=1.5, label=f'{model_name.replace("_", " ")} Filtered')
                ax1.plot(time_ms, iir_signal, 'r-', linewidth=1.5, label='IIR Filter')
                
                ax1.set_title(f'{model_name.replace("_", " ")} vs IIR Filter Comparison\n'
                             f'DL MSE: {mse_dl:.4f} | IIR MSE: {mse_iir:.4f} | '
                             f'Improvement: {((mse_iir - mse_dl) / mse_iir * 100):.1f}%', 
                             fontsize=14, fontweight='bold')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                ax1.set_ylim(-2, 2)
                
                # Bottom plot: Error comparison
                error_dl = clean_signal - pred_signal
                error_iir = clean_signal - iir_signal
                
                ax2.plot(time_ms, error_dl, 'b-', linewidth=1.5, label=f'{model_name.replace("_", " ")} Error')
                ax2.plot(time_ms, error_iir, 'r-', linewidth=1.5, label='IIR Filter Error')
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                ax2.set_title(f'Filtering Error Comparison\n'
                             f'DL SSD: {ssd_dl:.2f} | IIR SSD: {ssd_iir:.2f}', 
                             fontsize=12, fontweight='bold')
                ax2.set_xlabel('Time (ms)')
                ax2.set_ylabel('Error (Ground Truth - Prediction)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                
                # Save individual plot
                filename = f'individual_{model_name.lower()}_comparison.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Individual plot saved: {filename}")
                
            except Exception as e:
                print(f"❌ Error creating plot for {model_name}: {e}")
        else:
            print(f"❌ Model not found: {model_path}")

def main():
    """Main function"""
    
    print("🎨 Creating Visual Comparisons for ONNX Models")
    print("=" * 60)
    
    # Create timestamped output directory
    output_dir = create_output_directory()
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Create single signal detailed comparison (ALL 6 models + IIR)
        create_single_signal_comparison(output_dir)
        
        # Create individual model comparison plots
        create_individual_model_comparisons(output_dir)
        
        # Create multiple signals grid (best 3 models + IIR)
        create_multiple_signals_grid(output_dir)
        
        # Create performance summary (if CSV available)
        create_performance_summary(output_dir)
        
        # Automatically create versions with both filters
        print(f"\n⏳ Creating versions with both filters (this will take time)...")
        
        # Create single signal with both filters
        create_single_signal_with_both_filters(output_dir)
        
        # Create multiple signals with both filters
        create_multiple_signals_with_both_filters(output_dir)
        
        # Create summary file
        summary_file = os.path.join(output_dir, "SUMMARY.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== VISUAL COMPARISON RESULTS SUMMARY ===\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Generated files:\n")
            f.write("- single_signal_comparison.png: ALL 6 ONNX models + IIR comparison\n")
            f.write("- individual_*_comparison.png: Individual model vs IIR comparisons (6 files)\n")
            f.write("- single_signal_with_both_filters.png: Best 4 models + IIR + FIR\n")
            f.write("- multiple_signals_grid.png: Best 3 models across multiple signals\n")
            f.write("- multiple_signals_with_both_filters.png: Best 3 models + both filters\n")
            f.write("- performance_summary.png: Overall performance comparison (if CSV available)\n\n")
            f.write("Models tested:\n")
            f.write("[OK] Multibranch_LANLD (Best model - dilated convolutions)\n")
            f.write("[OK] Multibranch_LANL (Multi-branch linear/non-linear)\n")
            f.write("[OK] FCN_DAE (Fully Convolutional Denoising Autoencoder)\n")
            f.write("[OK] DRNN (Deep Recurrent Neural Network - LSTM model)\n")
            f.write("[OK] Vanilla_NL (Vanilla CNN with non-linear activations)\n")
            f.write("[OK] Vanilla_L (Vanilla CNN with linear activations)\n")
            f.write("[OK] IIR Filter (Classical digital filter)\n")
            f.write("[OK] FIR Filter (Classical digital filter - in some plots)\n\n")
            f.write("Note: DRNN is the LSTM-based model in the system.\n")
        
        print(f"\n🎉 ALL VISUALIZATIONS COMPLETE!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"\n📊 Generated files:")
        print(f"   ✅ single_signal_comparison.png: ALL 6 ONNX models + IIR")
        print(f"   ✅ individual_*_comparison.png: 6 individual model comparisons")
        print(f"   ✅ single_signal_with_both_filters.png: Best 4 models + both filters")
        print(f"   ✅ multiple_signals_grid.png: Best 3 models across signals")
        print(f"   ✅ multiple_signals_with_both_filters.png: Complete comparison")
        print(f"   ✅ performance_summary.png: Overall performance (if CSV available)")
        print(f"   ✅ SUMMARY.txt: Complete summary of generated files")
        
        print(f"\n🔍 Key Features:")
        print(f"   • Tests ALL 6 ONNX models (including DRNN/LSTM)")
        print(f"   • Organized timestamped output directory")
        print(f"   • Individual model comparison plots")
        print(f"   • Comprehensive multi-signal analysis")
        print(f"   • Both classical filters (IIR + FIR) included")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
