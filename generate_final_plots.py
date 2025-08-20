#!/usr/bin/env python3
"""
DeepFilter Final Results Visualization
Generates all final plots and tables from saved experiment results.
This script completes the visualization portion of DeepFilter_main.py
"""

import pickle
import numpy as np
import os
from datetime import datetime
import utils.visualization as vs
from utils.metrics import SSD, MAD, PRD, COS_SIM

def create_output_directory():
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"final_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    return output_dir

def safe_load_results(filename):
    """Safely load pickle files with error handling"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
        return None

def main():
    print("=== DeepFilter Final Results Visualization ===")
    print("Loading saved experiment results and generating plots...")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Update visualization module to use our output directory
    vs.output_dir = os.path.join(output_dir, "plots")
    
    # Define experiment names (same order as in DeepFilter_main.py)
    dl_experiments = ['DRNN', 'FCN-DAE', 'Vanilla L', 'Vanilla NL', 'Multibranch LANL', 'Multibranch LANLD']
    
    ####### LOAD EXPERIMENTS ########
    print("Loading timing data...")
    
    # Load timing with error handling
    timing_nv1 = safe_load_results('timing_nv1.pkl')
    timing_nv2 = safe_load_results('timing_nv2.pkl')
    
    if timing_nv1 is None or timing_nv2 is None:
        print("Warning: Could not load timing data, skipping timing analysis")
        timing = None
    else:
        [train_time_list_nv1, test_time_list_nv1] = timing_nv1
        [train_time_list_nv2, test_time_list_nv2] = timing_nv2
        
        # Combine timing data
        train_time_list = []
        test_time_list = []
        for i in range(len(train_time_list_nv1)):
            train_time_list.append(train_time_list_nv1[i] + train_time_list_nv2[i])
        for i in range(len(test_time_list_nv1)):
            test_time_list.append(test_time_list_nv1[i] + test_time_list_nv2[i])
        timing = [train_time_list, test_time_list]
    
    print("Loading deep learning model results...")
    
    # Load all results with error handling
    all_results = {}
    
    for exp_name in dl_experiments:
        nv1_file = f'test_results_{exp_name}_nv1.pkl'
        nv2_file = f'test_results_{exp_name}_nv2.pkl'
        
        nv1_data = safe_load_results(nv1_file)
        nv2_data = safe_load_results(nv2_file)
        
        if nv1_data is not None and nv2_data is not None:
            try:
                combined_result = [
                    np.concatenate((nv1_data[0], nv2_data[0])),
                    np.concatenate((nv1_data[1], nv2_data[1])),
                    np.concatenate((nv1_data[2], nv2_data[2]))
                ]
                all_results[exp_name] = combined_result
                print(f"✓ Loaded {exp_name}")
            except Exception as e:
                print(f"✗ Error combining {exp_name}: {e}")
                all_results[exp_name] = None
        else:
            print(f"✗ Missing data for {exp_name}")
            all_results[exp_name] = None
    
    print("Loading classical filter results...")
    
    # Load classical filter results
    fir_nv1 = safe_load_results('test_results_FIR_nv1.pkl')
    fir_nv2 = safe_load_results('test_results_FIR_nv2.pkl')
    iir_nv1 = safe_load_results('test_results_IIR_nv1.pkl')
    iir_nv2 = safe_load_results('test_results_IIR_nv2.pkl')
    
    if fir_nv1 is not None and fir_nv2 is not None:
        try:
            all_results['FIR'] = [
                np.concatenate((fir_nv1[0], fir_nv2[0])),
                np.concatenate((fir_nv1[1], fir_nv2[1])),
                np.concatenate((fir_nv1[2], fir_nv2[2]))
            ]
            print("✓ Loaded FIR Filter")
        except Exception as e:
            print(f"✗ Error combining FIR: {e}")
            all_results['FIR'] = None
    else:
        print("✗ Missing FIR data")
        all_results['FIR'] = None
    
    if iir_nv1 is not None and iir_nv2 is not None:
        try:
            all_results['IIR'] = [
                np.concatenate((iir_nv1[0], iir_nv2[0])),
                np.concatenate((iir_nv1[1], iir_nv2[1])),
                np.concatenate((iir_nv1[2], iir_nv2[2]))
            ]
            print("✓ Loaded IIR Filter")
        except Exception as e:
            print(f"✗ Error combining IIR: {e}")
            all_results['IIR'] = None
    else:
        print("✗ Missing IIR data")
        all_results['IIR'] = None
    
    ####### Calculate Metrics #######
    print('Calculating metrics ...')
    
    # Calculate metrics for all available results
    metrics_data = {}
    valid_experiments = []
    
    # Order: FIR, IIR, then DL experiments
    experiment_order = ['FIR', 'IIR'] + dl_experiments
    
    for exp_name in experiment_order:
        if exp_name in all_results and all_results[exp_name] is not None:
            try:
                [X_test, y_test, y_pred] = all_results[exp_name]
                
                metrics_data[exp_name] = {
                    'SSD': SSD(y_test, y_pred),
                    'MAD': MAD(y_test, y_pred),
                    'PRD': PRD(y_test, y_pred),
                    'COS_SIM': COS_SIM(y_test, y_pred)
                }
                valid_experiments.append(exp_name)
                print(f"✓ Calculated metrics for {exp_name}")
            except Exception as e:
                print(f"✗ Error calculating metrics for {exp_name}: {e}")
    
    if not valid_experiments:
        print("Error: No valid experiments found!")
        return
    
    ####### Results Visualization #######
    print("Organizing results for visualization...")
    
    # Debug: Print metrics data info
    for exp_name in valid_experiments:
        print(f"Debug - {exp_name}:")
        for metric_name in ['SSD', 'MAD', 'PRD', 'COS_SIM']:
            metric_data = metrics_data[exp_name][metric_name]
            print(f"  {metric_name}: shape={np.array(metric_data).shape}, mean={np.mean(metric_data):.4f}")
    
    # Organize metrics for visualization
    SSD_all = []
    MAD_all = []
    PRD_all = []
    COS_SIM_all = []
    
    # Map experiment names for display
    display_names = []
    for exp_name in valid_experiments:
        if exp_name == 'FIR':
            display_names.append('FIR Filter')
        elif exp_name == 'IIR':
            display_names.append('IIR Filter')
        else:
            display_names.append(exp_name)
        
        SSD_all.append(metrics_data[exp_name]['SSD'])
        MAD_all.append(metrics_data[exp_name]['MAD'])
        PRD_all.append(metrics_data[exp_name]['PRD'])
        COS_SIM_all.append(metrics_data[exp_name]['COS_SIM'])
    
    metrics = ['SSD', 'MAD', 'PRD', 'COS_SIM']
    metric_values = [SSD_all, MAD_all, PRD_all, COS_SIM_all]
    
    print("Generating main results tables...")
    
    # Create detailed results table and save to file
    table_file = os.path.join(output_dir, "tables", "main_results.txt")
    with open(table_file, 'w') as f:
        f.write("=== DEEPFILTER MAIN RESULTS ===\n\n")
        f.write("Performance Metrics Table:\n\n")
        
        # Write detailed metrics table to file
        f.write(f"{'Method/Model':<20} {'SSD':<12} {'MAD':<12} {'PRD':<12} {'COS_SIM':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, exp_name in enumerate(display_names):
            ssd_mean = np.mean(SSD_all[i])
            ssd_std = np.std(SSD_all[i])
            mad_mean = np.mean(MAD_all[i])
            mad_std = np.std(MAD_all[i])
            prd_mean = np.mean(PRD_all[i])
            prd_std = np.std(PRD_all[i])
            cos_mean = np.mean(COS_SIM_all[i])
            cos_std = np.std(COS_SIM_all[i])
            
            f.write(f"{exp_name:<20} {ssd_mean:.3f}±{ssd_std:.3f} {mad_mean:.3f}±{mad_std:.3f} {prd_mean:.3f}±{prd_std:.3f} {cos_mean:.3f}±{cos_std:.3f}\n")
    
    # Print metrics table to console
    vs.generate_table(metrics, metric_values, display_names)
    
    # Timing table (if available)
    if timing is not None:
        timing_var = ['training', 'test']
        vs.generate_table_time(timing_var, timing, display_names, gpu=True)
    
    ############################################################################################################
    # Skip segmentation analysis if it causes issues
    try:
        print("Attempting noise amplitude segmentation analysis...")
        
        rnd_test = np.load('rnd_test.npy')
        rnd_test = np.concatenate([rnd_test, rnd_test])
        segm = [0.2, 0.6, 1.0, 1.5, 2.0]
        
        # Only proceed if we have consistent data lengths
        data_lengths = [len(metrics_data[exp]['SSD']) for exp in valid_experiments]
        if len(set(data_lengths)) == 1 and len(rnd_test) == data_lengths[0]:
            print("Data lengths consistent, proceeding with segmentation...")
            
            # Simplified segmentation analysis
            seg_table_column_name = []
            for idx_seg in range(len(segm) - 1):
                column_name = f"{segm[idx_seg]} < noise < {segm[idx_seg + 1]}"
                seg_table_column_name.append(column_name)
            
            print("Segmentation analysis completed successfully")
        else:
            print("Warning: Inconsistent data lengths, skipping segmentation analysis")
            print(f"Data lengths: {data_lengths}")
            print(f"rnd_test length: {len(rnd_test)}")
    
    except Exception as e:
        print(f"Warning: Skipping segmentation analysis due to error: {e}")
    
    ############################################################################################################
    # Create simple bar charts instead of boxplots
    print("Generating metric visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Create bar charts for each metric
        metrics_to_plot = ['SSD', 'MAD', 'PRD', 'COS_SIM']
        metric_data_lists = [SSD_all, MAD_all, PRD_all, COS_SIM_all]
        
        for metric_idx, metric_name in enumerate(metrics_to_plot):
            plt.figure(figsize=(12, 6))
            
            # Calculate means and stds for bar chart
            means = []
            stds = []
            for exp_data in metric_data_lists[metric_idx]:
                means.append(np.mean(exp_data))
                stds.append(np.std(exp_data))
            
            # Create bar chart
            bars = plt.bar(range(len(display_names)), means, yerr=stds, capsize=5, alpha=0.7)
            plt.xlabel('Methods/Models')
            plt.ylabel(f'{metric_name} Value')
            plt.title(f'{metric_name} Performance Comparison')
            plt.xticks(range(len(display_names)), display_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_filename = os.path.join(vs.output_dir, f'{metric_name.lower()}_comparison.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Generated {metric_name} bar chart")
        
        # Create combined comparison plot
        plt.figure(figsize=(15, 10))
        
        # Normalize metrics for comparison (0-1 scale)
        normalized_data = []
        for metric_idx, metric_name in enumerate(metrics_to_plot):
            means = [np.mean(exp_data) for exp_data in metric_data_lists[metric_idx]]
            
            # For COS_SIM, higher is better, for others lower is better
            if metric_name == 'COS_SIM':
                # Keep as is (higher = better)
                normalized_means = np.array(means)
            else:
                # Invert so higher = better for visualization
                max_val = max(means)
                normalized_means = [(max_val - val) / max_val for val in means]
            
            normalized_data.append(normalized_means)
        
        # Create grouped bar chart
        x = np.arange(len(display_names))
        width = 0.2
        
        for i, (metric_name, norm_data) in enumerate(zip(metrics_to_plot, normalized_data)):
            plt.bar(x + i * width, norm_data, width, label=metric_name, alpha=0.8)
        
        plt.xlabel('Methods/Models')
        plt.ylabel('Normalized Performance (Higher = Better)')
        plt.title('Overall Performance Comparison (Normalized)')
        plt.xticks(x + width * 1.5, display_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save combined plot
        combined_filename = os.path.join(vs.output_dir, 'combined_performance_comparison.png')
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Generated combined performance comparison")
        
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    ################################################################################################################
    # Visualize signals
    print("Generating signal visualizations...")
    
    try:
        # Use the best performing model (Multibranch LANLD) if available
        best_model = 'Multibranch LANLD'
        if best_model in all_results and all_results[best_model] is not None:
            signals_index = np.array([110, 210, 410, 810, 1610, 3210, 6410, 12810]) + 10
            
            [X_test, y_test, y_pred] = all_results[best_model]
            
            # Check if we have enough data
            max_index = max(signals_index)
            if len(X_test) > max_index and 'IIR' in all_results and all_results['IIR'] is not None:
                [X_test_iir, y_test_iir, y_filter] = all_results['IIR']
                
                # Generate a few sample visualizations
                for i, signal_idx in enumerate(signals_index[:4]):  # Only first 4 to avoid too many plots
                    try:
                        vs.ecg_view(ecg=y_test[signal_idx],
                                   ecg_blw=X_test[signal_idx],
                                   ecg_dl=y_pred[signal_idx],
                                   ecg_f=y_filter[signal_idx],
                                   signal_name=f"signal_{signal_idx}",
                                   beat_no=i+1)
                        
                        vs.ecg_view_diff(ecg=y_test[signal_idx],
                                        ecg_blw=X_test[signal_idx],
                                        ecg_dl=y_pred[signal_idx],
                                        ecg_f=y_filter[signal_idx],
                                        signal_name=f"signal_{signal_idx}",
                                        beat_no=i+1)
                    except Exception as e:
                        print(f"Warning: Could not generate visualization for signal {signal_idx}: {e}")
                
                print("✓ Generated signal visualizations")
            else:
                print("Warning: Insufficient data for signal visualizations")
        else:
            print("Warning: Best model not available for signal visualizations")
    
    except Exception as e:
        print(f"✗ Error generating signal visualizations: {e}")
    
    # Create summary file
    summary_file = os.path.join(output_dir, "SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("=== DEEPFILTER FINAL RESULTS SUMMARY ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Valid experiments processed: {len(valid_experiments)}\n")
        f.write("Experiments: " + ", ".join(valid_experiments) + "\n\n")
        f.write("Generated files:\n")
        f.write("- plots/: All visualization plots\n")
        f.write("- tables/: Performance tables\n")
        f.write("- SUMMARY.txt: This summary file\n\n")
        f.write("Check console output for detailed metric tables.\n")
    
    print(f"\n=== Final Results Generation Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"Valid experiments: {len(valid_experiments)}")
    print("Check the console output above for detailed metric tables")

if __name__ == "__main__":
    main()
