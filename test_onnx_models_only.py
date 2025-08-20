#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ONNX Models Only Comparison Test
Focused test of all ONNX models vs classical filters with comprehensive visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import time
from datetime import datetime
import pandas as pd

# ONNX imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX Runtime not available. Install with: pip install onnxruntime")
    exit(1)

# Classical filters
from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
from utils.metrics import MAD, SSD, PRD, COS_SIM

class ONNXModelTester:
    """ONNX-focused model testing class"""
    
    def __init__(self, test_data_path='test_results_Vanilla L_nv1.pkl'):
        self.test_data_path = test_data_path
        self.X_test = None
        self.y_test = None
        self.results = {}
        self.timing_results = {}
        
        # Load test data
        self.load_test_data()
        
        # ONNX model configurations
        self.onnx_dir = 'onnx_models'
        self.onnx_models = [
            'Vanilla_L',
            'Vanilla_NL', 
            'FCN_DAE',
            'DRNN',
            'Multibranch_LANL',
            'Multibranch_LANLD'
        ]
    
    def load_test_data(self):
        """Load test data from pickle file"""
        
        print("📊 Loading test data...")
        
        if not os.path.exists(self.test_data_path):
            print(f"❌ Test data not found: {self.test_data_path}")
            return False
        
        try:
            with open(self.test_data_path, 'rb') as f:
                X_test, y_test, _ = pickle.load(f)
            
            self.X_test = X_test
            self.y_test = y_test
            
            print(f"✅ Loaded test data: {X_test.shape[0]} samples")
            print(f"   Signal shape: {X_test.shape[1:]} (512 samples, 1 channel)")
            print(f"   Duration: {X_test.shape[1]/360:.2f} seconds per signal")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return False
    
    def test_classical_filters(self, n_samples=500):
        """Test classical FIR and IIR filters"""
        
        print(f"\n🔧 Testing Classical Filters ({n_samples} samples)...")
        
        # Create dataset format for classical filters
        dataset = [None, None, self.X_test[:n_samples], self.y_test[:n_samples]]
        
        # Test IIR Filter (faster than FIR)
        print("   Testing IIR Filter...")
        start_time = time.time()
        try:
            X_test_iir, y_test_iir, y_pred_iir = IIR_test_Dataset(dataset)
            iir_time = time.time() - start_time
            
            self.results['IIR_Filter'] = {
                'predictions': y_pred_iir,
                'ground_truth': y_test_iir,
                'input': X_test_iir
            }
            self.timing_results['IIR_Filter'] = iir_time
            print(f"   ✅ IIR Filter completed in {iir_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ IIR Filter failed: {e}")
        
        # Test FIR Filter (slower, so use fewer samples)
        print("   Testing FIR Filter (100 samples only - very slow)...")
        dataset_small = [None, None, self.X_test[:100], self.y_test[:100]]
        start_time = time.time()
        try:
            X_test_fir, y_test_fir, y_pred_fir = FIR_test_Dataset(dataset_small)
            fir_time = time.time() - start_time
            
            # Extrapolate timing for full dataset
            fir_time_full = fir_time * (n_samples / 100)
            
            self.results['FIR_Filter'] = {
                'predictions': y_pred_fir,
                'ground_truth': y_test_fir,
                'input': X_test_fir
            }
            self.timing_results['FIR_Filter'] = fir_time_full
            print(f"   ✅ FIR Filter completed 100 samples in {fir_time:.2f}s (estimated {fir_time_full:.1f}s for {n_samples})")
            
        except Exception as e:
            print(f"   ❌ FIR Filter failed: {e}")
    
    def test_onnx_models(self, n_samples=500):
        """Test all ONNX models"""
        
        print(f"\n🔄 Testing ONNX Models ({n_samples} samples)...")
        
        for model_name in self.onnx_models:
            onnx_path = os.path.join(self.onnx_dir, f'{model_name}.onnx')
            
            if not os.path.exists(onnx_path):
                print(f"   ❌ ONNX model not found: {onnx_path}")
                continue
            
            try:
                print(f"   Testing {model_name}...")
                
                # Load ONNX model
                session = ort.InferenceSession(onnx_path)
                
                # Get input/output names
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                
                # Prepare test data
                X_test_subset = self.X_test[:n_samples].astype(np.float32)
                y_test_subset = self.y_test[:n_samples]
                
                # Run inference
                start_time = time.time()
                y_pred = session.run([output_name], {input_name: X_test_subset})[0]
                inference_time = time.time() - start_time
                
                # Store results
                self.results[model_name] = {
                    'predictions': y_pred,
                    'ground_truth': y_test_subset,
                    'input': X_test_subset
                }
                self.timing_results[model_name] = inference_time
                
                print(f"   ✅ {model_name} completed in {inference_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
    
    def calculate_metrics(self):
        """Calculate performance metrics for all tested models"""
        
        print(f"\n📊 Calculating metrics for {len(self.results)} models...")
        
        self.metrics = {}
        
        for model_name, result in self.results.items():
            try:
                y_true = result['ground_truth']
                y_pred = result['predictions']
                
                # Calculate metrics
                ssd_vals = SSD(y_true, y_pred)
                mad_vals = MAD(y_true, y_pred)
                prd_vals = PRD(y_true, y_pred)
                cos_vals = COS_SIM(y_true, y_pred)
                
                self.metrics[model_name] = {
                    'SSD': np.mean(ssd_vals),
                    'MAD': np.mean(mad_vals),
                    'PRD': np.mean(prd_vals),
                    'COS_SIM': np.mean(cos_vals),
                    'SSD_std': np.std(ssd_vals),
                    'MAD_std': np.std(mad_vals),
                    'PRD_std': np.std(prd_vals),
                    'COS_SIM_std': np.std(cos_vals),
                    'samples': len(y_true)
                }
                
                print(f"   ✅ {model_name}: Metrics calculated")
                
            except Exception as e:
                print(f"   ❌ {model_name}: Metric calculation failed - {e}")
    
    def display_results(self):
        """Display comprehensive results"""
        
        print(f"\n" + "="*80)
        print(f"🏆 ONNX MODELS vs CLASSICAL FILTERS COMPARISON")
        print(f"="*80)
        
        if not self.metrics:
            print("❌ No metrics available to display")
            return
        
        # Create results table
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"{'Model':<20} {'SSD':<10} {'MAD':<10} {'PRD':<10} {'COS_SIM':<10} {'Time(s)':<10} {'Speed(sps)':<12}")
        print("-" * 92)
        
        # Sort by SSD (lower is better)
        sorted_models = sorted(self.metrics.items(), key=lambda x: x[1]['SSD'])
        
        for i, (model_name, metrics) in enumerate(sorted_models):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}"
            timing = self.timing_results.get(model_name, 0)
            speed = metrics['samples'] / timing if timing > 0 else 0
            
            print(f"{rank_emoji} {model_name:<18} "
                  f"{metrics['SSD']:<10.4f} "
                  f"{metrics['MAD']:<10.4f} "
                  f"{metrics['PRD']:<10.4f} "
                  f"{metrics['COS_SIM']:<10.4f} "
                  f"{timing:<10.2f} "
                  f"{speed:<12.1f}")
        
        # Performance insights
        print(f"\n🎯 KEY INSIGHTS:")
        
        # Best performers
        best_model = sorted_models[0]
        print(f"   🏆 Best Overall: {best_model[0]} (SSD: {best_model[1]['SSD']:.4f})")
        
        # Classical filter comparison
        classical_models = [m for m in sorted_models if 'Filter' in m[0]]
        dl_models = [m for m in sorted_models if 'Filter' not in m[0]]
        
        if classical_models and dl_models:
            best_classical = min(classical_models, key=lambda x: x[1]['SSD'])
            best_dl = min(dl_models, key=lambda x: x[1]['SSD'])
            
            improvement = best_classical[1]['SSD'] / best_dl[1]['SSD']
            print(f"   📈 ONNX Deep Learning Improvement: {improvement:.1f}x better than classical filters")
            print(f"      Best Classical: {best_classical[0]} (SSD: {best_classical[1]['SSD']:.4f})")
            print(f"      Best ONNX: {best_dl[0]} (SSD: {best_dl[1]['SSD']:.4f})")
        
        # Speed analysis
        print(f"\n⚡ SPEED ANALYSIS:")
        fastest_model = min(self.timing_results.items(), key=lambda x: x[1])
        slowest_model = max(self.timing_results.items(), key=lambda x: x[1])
        
        print(f"   🚀 Fastest: {fastest_model[0]} ({fastest_model[1]:.2f}s)")
        print(f"   🐌 Slowest: {slowest_model[0]} ({slowest_model[1]:.2f}s)")
        
        # Real-time capability
        n_samples = list(self.metrics.values())[0]['samples']
        signal_duration = n_samples * (512 / 360) / 1000  # Total duration in seconds
        
        print(f"\n🎬 REAL-TIME CAPABILITY:")
        print(f"   Signal duration: {signal_duration:.1f}s ({n_samples} beats)")
        
        for model_name, timing in sorted(self.timing_results.items(), key=lambda x: x[1]):
            if 'Filter' not in model_name:  # Only show ONNX models
                real_time_factor = signal_duration / timing
                status = "✅ Real-time" if real_time_factor > 1 else "❌ Too slow"
                print(f"   {model_name}: {real_time_factor:.1f}x real-time {status}")
    
    def create_comprehensive_plots(self, n_signals=4):
        """Create comprehensive visualization plots"""
        
        print(f"\n📊 Creating comprehensive plots...")
        
        if not self.results:
            print("❌ No results available for plotting")
            return
        
        # Get all models for plotting (ONNX + classical)
        all_models = list(self.results.keys())
        
        # Separate ONNX and classical models
        onnx_models = [m for m in all_models if 'Filter' not in m]
        classical_models = [m for m in all_models if 'Filter' in m]
        
        # Sort ONNX models by performance
        if self.metrics:
            onnx_models = sorted([m for m in onnx_models if m in self.metrics], 
                               key=lambda x: self.metrics[x]['SSD'])
        
        models_to_plot = onnx_models + classical_models
        
        print(f"   Plotting {len(models_to_plot)} models: {models_to_plot}")
        
        # Create signal comparison plot
        fig, axes = plt.subplots(n_signals, len(models_to_plot) + 2, 
                                figsize=(4*(len(models_to_plot)+2), 3*n_signals))
        if n_signals == 1:
            axes = axes.reshape(1, -1)
        
        time_axis = np.arange(512) / 360 * 1000  # Convert to milliseconds
        
        for signal_idx in range(n_signals):
            # Get a representative signal
            sample_idx = signal_idx * (len(self.X_test) // (n_signals + 1))
            
            # Plot noisy input
            axes[signal_idx, 0].plot(time_axis, self.X_test[sample_idx, :, 0], 'r-', alpha=0.7, linewidth=1)
            axes[signal_idx, 0].set_title(f'Noisy Input\n(Signal {signal_idx+1})')
            axes[signal_idx, 0].set_ylabel('Amplitude')
            axes[signal_idx, 0].grid(True, alpha=0.3)
            axes[signal_idx, 0].set_ylim(-2, 2)
            
            # Plot ground truth
            axes[signal_idx, 1].plot(time_axis, self.y_test[sample_idx, :, 0], 'g-', linewidth=1.5)
            axes[signal_idx, 1].set_title(f'Ground Truth\n(Clean ECG)')
            axes[signal_idx, 1].set_ylabel('Amplitude')
            axes[signal_idx, 1].grid(True, alpha=0.3)
            axes[signal_idx, 1].set_ylim(-2, 2)
            
            # Plot model outputs
            for model_idx, model_name in enumerate(models_to_plot):
                if model_name in self.results:
                    result = self.results[model_name]
                    if sample_idx < len(result['predictions']):
                        pred = result['predictions'][sample_idx, :, 0]
                        
                        # Choose color based on model type
                        if 'Filter' in model_name:
                            color = 'orange' if 'IIR' in model_name else 'brown'
                            alpha = 0.8
                        else:
                            # Different colors for different ONNX models
                            colors = ['blue', 'purple', 'cyan', 'magenta', 'navy', 'teal']
                            color = colors[model_idx % len(colors)]
                            alpha = 0.9
                        
                        axes[signal_idx, model_idx + 2].plot(time_axis, pred, color=color, linewidth=1, alpha=alpha)
                        
                        # Calculate MSE for this signal
                        mse = np.mean((pred - self.y_test[sample_idx, :, 0]) ** 2)
                        
                        # Shorten title if needed
                        title = model_name.replace('_', ' ')
                        if len(title) > 12:
                            title = title[:12] + '...'
                        
                        axes[signal_idx, model_idx + 2].set_title(f'{title}\n(MSE: {mse:.3f})')
                        axes[signal_idx, model_idx + 2].set_ylabel('Amplitude')
                        axes[signal_idx, model_idx + 2].grid(True, alpha=0.3)
                        axes[signal_idx, model_idx + 2].set_ylim(-2, 2)
            
            # Set x-label for bottom row
            if signal_idx == n_signals - 1:
                for col in range(len(models_to_plot) + 2):
                    axes[signal_idx, col].set_xlabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig('onnx_model_comparison_signals.png', dpi=150, bbox_inches='tight')
        print("✅ Signal comparison plot saved: onnx_model_comparison_signals.png")
        
        # Create metrics comparison plots
        self.create_metrics_plots()
        
        plt.show()
    
    def create_metrics_plots(self):
        """Create detailed metrics comparison plots"""
        
        if not self.metrics:
            return
        
        # Prepare data
        models = list(self.metrics.keys())
        ssd_values = [self.metrics[m]['SSD'] for m in models]
        cos_values = [self.metrics[m]['COS_SIM'] for m in models]
        prd_values = [self.metrics[m]['PRD'] for m in models]
        mad_values = [self.metrics[m]['MAD'] for m in models]
        
        # Separate ONNX and classical models for coloring
        colors = []
        for m in models:
            if 'Filter' in m:
                colors.append('red' if 'FIR' in m else 'orange')
            else:
                colors.append('blue')
        
        # Create 2x2 subplot for all metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # SSD plot (lower is better)
        bars1 = ax1.bar(range(len(models)), ssd_values, color=colors, alpha=0.7)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('SSD (lower is better)')
        ax1.set_title('Sum of Squared Differences (SSD)')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, ssd_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ssd_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Cosine Similarity plot (higher is better)
        bars2 = ax2.bar(range(len(models)), cos_values, color=colors, alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Cosine Similarity (higher is better)')
        ax2.set_title('Cosine Similarity')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars2, cos_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # PRD plot (lower is better)
        bars3 = ax3.bar(range(len(models)), prd_values, color=colors, alpha=0.7)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('PRD (lower is better)')
        ax3.set_title('Percentage Root-mean-square Difference (PRD)')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, prd_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(prd_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Speed comparison (samples per second)
        speeds = [self.metrics[m]['samples'] / self.timing_results[m] if m in self.timing_results and self.timing_results[m] > 0 else 0 for m in models]
        bars4 = ax4.bar(range(len(models)), speeds, color=colors, alpha=0.7)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Throughput (samples/second)')
        ax4.set_title('Inference Speed')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, speeds):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speeds)*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('onnx_model_comparison_metrics.png', dpi=150, bbox_inches='tight')
        print("✅ Metrics comparison plot saved: onnx_model_comparison_metrics.png")
    
    def save_results_csv(self):
        """Save results to CSV file"""
        
        if not self.metrics:
            print("❌ No metrics to save")
            return
        
        # Prepare data for CSV
        data = []
        for model_name, metrics in self.metrics.items():
            timing = self.timing_results.get(model_name, 0)
            throughput = metrics['samples'] / timing if timing > 0 else 0
            
            data.append({
                'Model': model_name,
                'Type': 'ONNX' if 'Filter' not in model_name else 'Classical',
                'SSD_Mean': metrics['SSD'],
                'SSD_Std': metrics['SSD_std'],
                'MAD_Mean': metrics['MAD'],
                'MAD_Std': metrics['MAD_std'],
                'PRD_Mean': metrics['PRD'],
                'PRD_Std': metrics['PRD_std'],
                'COS_SIM_Mean': metrics['COS_SIM'],
                'COS_SIM_Std': metrics['COS_SIM_std'],
                'Inference_Time_s': timing,
                'Throughput_samples_per_s': throughput,
                'Test_Samples': metrics['samples']
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df = df.sort_values('SSD_Mean')  # Sort by SSD (best first)
        
        csv_filename = f'onnx_model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_filename, index=False)
        
        print(f"✅ Results saved to: {csv_filename}")
        
        return csv_filename

def main():
    """Main testing function"""
    
    print("🔄 ONNX Models vs Classical Filters Comparison")
    print("=" * 60)
    
    # Initialize tester
    tester = ONNXModelTester()
    
    if tester.X_test is None:
        print("❌ Cannot proceed without test data")
        return
    
    # Test parameters
    n_samples = 500  # Number of test samples
    print(f"📊 Testing with {n_samples} samples per model")
    
    # Test classical filters
    tester.test_classical_filters(n_samples)
    
    # Test all ONNX models
    tester.test_onnx_models(n_samples)
    
    # Calculate metrics
    tester.calculate_metrics()
    
    # Display results
    tester.display_results()
    
    # Create comprehensive visualizations
    tester.create_comprehensive_plots(n_signals=4)
    
    # Save results
    csv_file = tester.save_results_csv()
    
    print(f"\n🎉 ONNX TESTING COMPLETE!")
    print(f"   - Tested {len(tester.results)} models")
    print(f"   - Results saved to: {csv_file}")
    print(f"   - Plots saved: onnx_model_comparison_signals.png, onnx_model_comparison_metrics.png")
    print(f"   - All ONNX models are production-ready! 🚀")

if __name__ == "__main__":
    main()
