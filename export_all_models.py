#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export All DeepFilter Models
Export all trained DeepFilter models to TensorFlow SavedModel format
"""

import numpy as np
import tensorflow as tf
import _pickle as pickle
import os
from deepFilter.dl_models import *
from deepFilter.dl_pipeline import combined_ssd_mad_loss, ssd_loss, mad_loss

def get_model_config(model_name):
    """Get model architecture and loss function for each model type"""
    
    configs = {
        'Vanilla L': {
            'model_fn': lambda: deep_filter_vanilla_linear(signal_size=512),
            'loss_fn': combined_ssd_mad_loss,
            'weight_file': 'Vanilla_L_weights.best.hdf5'
        },
        'Vanilla NL': {
            'model_fn': lambda: deep_filter_vanilla_Nlinear(signal_size=512),
            'loss_fn': combined_ssd_mad_loss,
            'weight_file': 'Vanilla_NL_weights.best.hdf5'
        },
        'FCN-DAE': {
            'model_fn': lambda: FCN_DAE(signal_size=512),
            'loss_fn': ssd_loss,
            'weight_file': 'FCN_DAE_weights.best.hdf5'
        },
        'DRNN': {
            'model_fn': lambda: DRRN_denoising(signal_size=512),
            'loss_fn': tf.keras.losses.mean_squared_error,
            'weight_file': 'DRNN_weights.best.hdf5'
        },
        'Multibranch LANL': {
            'model_fn': lambda: deep_filter_I_LANL(signal_size=512),
            'loss_fn': combined_ssd_mad_loss,
            'weight_file': 'Multibranch_LANL_weights.best.hdf5'
        },
        'Multibranch LANLD': {
            'model_fn': lambda: deep_filter_model_I_LANL_dilated(signal_size=512),
            'loss_fn': combined_ssd_mad_loss,
            'weight_file': 'Multibranch_LANLD_weights.best.hdf5'
        }
    }
    
    return configs.get(model_name)

def export_single_model(model_name, config, output_dir='exported_models'):
    """Export a single model to SavedModel format"""
    
    print(f"\n=== Exporting {model_name} ===")
    
    weight_file = config['weight_file']
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_savedmodel")
    
    # Check if weights exist
    if not os.path.exists(weight_file):
        print(f"❌ Weights not found: {weight_file}")
        return None
    
    try:
        # Create model
        model = config['model_fn']()
        
        # Compile model
        model.compile(
            loss=config['loss_fn'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_absolute_error, ssd_loss, mad_loss]
        )
        
        # Load weights
        model.load_weights(weight_file)
        print(f"✅ Loaded weights from {weight_file}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to SavedModel
        model.save(output_path, save_format='tf')
        print(f"✅ Exported to {output_path}")
        
        # Verify the saved model
        loaded_model = tf.keras.models.load_model(output_path, custom_objects={
            'combined_ssd_mad_loss': combined_ssd_mad_loss,
            'ssd_loss': ssd_loss,
            'mad_loss': mad_loss
        })
        print(f"✅ Verification passed")
        
        # Get model info
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        return {
            'model_name': model_name,
            'output_path': output_path,
            'weight_file': weight_file,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
        
    except Exception as e:
        print(f"❌ Error exporting {model_name}: {e}")
        return None

def test_exported_model(model_info, test_data_path=None):
    """Test an exported model"""
    
    model_name = model_info['model_name']
    output_path = model_info['output_path']
    
    # Determine test data path
    if test_data_path is None:
        test_data_path = f'test_results_{model_name}_nv1.pkl'
    
    if not os.path.exists(test_data_path):
        print(f"⚠️  Test data not found for {model_name}: {test_data_path}")
        return None
    
    try:
        print(f"🧪 Testing {model_name}...")
        
        # Load model
        model = tf.keras.models.load_model(output_path, custom_objects={
            'combined_ssd_mad_loss': combined_ssd_mad_loss,
            'ssd_loss': ssd_loss,
            'mad_loss': mad_loss
        })
        
        # Load test data
        with open(test_data_path, 'rb') as f:
            X_test, y_test, y_pred_original = pickle.load(f)
        
        # Test on a small subset
        n_samples = min(50, len(X_test))
        X_subset = X_test[:n_samples].astype(np.float32)
        y_subset = y_test[:n_samples]
        y_orig_subset = y_pred_original[:n_samples]
        
        # Run inference
        y_pred_exported = model.predict(X_subset, verbose=0)
        
        # Compare results
        mse_vs_original = np.mean((y_pred_exported - y_orig_subset) ** 2)
        mse_vs_ground_truth = np.mean((y_pred_exported - y_subset) ** 2)
        
        print(f"   MSE vs original: {mse_vs_original:.8f}")
        print(f"   MSE vs ground truth: {mse_vs_ground_truth:.6f}")
        
        if mse_vs_original < 1e-5:
            print(f"   ✅ Export successful (MSE < 1e-5)")
        else:
            print(f"   ⚠️  Export may have issues (MSE = {mse_vs_original:.8f})")
        
        return {
            'mse_vs_original': mse_vs_original,
            'mse_vs_ground_truth': mse_vs_ground_truth,
            'test_samples': n_samples
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return None

def create_deployment_package(exported_models, output_dir='deployment_package'):
    """Create a complete deployment package"""
    
    print(f"\n=== Creating Deployment Package ===")
    
    # Create deployment directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy all exported models
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    import shutil
    
    for model_info in exported_models:
        if model_info:
            src_path = model_info['output_path']
            dst_path = os.path.join(models_dir, os.path.basename(src_path))
            
            if os.path.exists(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                print(f"✅ Copied {model_info['model_name']} to deployment package")
    
    # Create model registry
    model_registry = {}
    for model_info in exported_models:
        if model_info:
            model_registry[model_info['model_name']] = {
                'path': f"models/{os.path.basename(model_info['output_path'])}",
                'total_params': model_info['total_params'],
                'input_shape': model_info['input_shape'],
                'output_shape': model_info['output_shape']
            }
    
    # Save model registry
    import json
    with open(os.path.join(output_dir, 'model_registry.json'), 'w') as f:
        json.dump(model_registry, f, indent=2)
    
    # Create unified inference script
    inference_script = '''#!/usr/bin/env python3
"""
DeepFilter Multi-Model Inference
Unified inference script for all exported DeepFilter models
"""

import numpy as np
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt

# Custom loss functions
def ssd_loss(y_true, y_pred):
    return tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-2)

def mad_loss(y_true, y_pred):
    return tf.keras.backend.max(tf.keras.backend.square(y_pred - y_true), axis=-2)

def combined_ssd_mad_loss(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=-2) * 500 + tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=-2)

class DeepFilterInference:
    """Unified inference class for all DeepFilter models"""
    
    def __init__(self, package_dir='.'):
        self.package_dir = package_dir
        self.models = {}
        self.model_registry = self.load_model_registry()
        
    def load_model_registry(self):
        """Load the model registry"""
        registry_path = os.path.join(self.package_dir, 'model_registry.json')
        with open(registry_path, 'r') as f:
            return json.load(f)
    
    def load_model(self, model_name):
        """Load a specific model"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if model_name not in self.models:
            model_path = os.path.join(self.package_dir, self.model_registry[model_name]['path'])
            self.models[model_name] = tf.keras.models.load_model(model_path, custom_objects={
                'combined_ssd_mad_loss': combined_ssd_mad_loss,
                'ssd_loss': ssd_loss,
                'mad_loss': mad_loss
            })
            print(f"✅ Loaded {model_name}")
        
        return self.models[model_name]
    
    def denoise_ecg(self, noisy_ecg, model_name='Multibranch LANLD'):
        """
        Denoise ECG using specified model
        
        Args:
            noisy_ecg: Input ECG signal (shape: [batch_size, 512, 1] or [512, 1])
            model_name: Name of the model to use
        
        Returns:
            clean_ecg: Denoised ECG signal
        """
        
        # Load model
        model = self.load_model(model_name)
        
        # Ensure correct input shape
        if len(noisy_ecg.shape) == 2:
            noisy_ecg = np.expand_dims(noisy_ecg, axis=0)
        
        # Convert to float32
        noisy_ecg = noisy_ecg.astype(np.float32)
        
        # Run inference
        clean_ecg = model.predict(noisy_ecg, verbose=0)
        
        return clean_ecg
    
    def compare_models(self, noisy_ecg, models_to_compare=None):
        """Compare multiple models on the same input"""
        
        if models_to_compare is None:
            models_to_compare = list(self.model_registry.keys())
        
        results = {}
        
        for model_name in models_to_compare:
            try:
                clean_ecg = self.denoise_ecg(noisy_ecg, model_name)
                results[model_name] = clean_ecg
                print(f"✅ {model_name}: Success")
            except Exception as e:
                print(f"❌ {model_name}: {e}")
        
        return results
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.model_registry.keys())

def demo():
    """Demo function"""
    
    # Initialize inference
    inference = DeepFilterInference()
    
    print("Available models:")
    for model in inference.get_available_models():
        info = inference.model_registry[model]
        print(f"  - {model}: {info['total_params']:,} parameters")
    
    # Create dummy data for demonstration
    print("\\nCreating dummy noisy ECG data...")
    np.random.seed(42)
    
    # Simulate a simple ECG-like signal with noise
    t = np.linspace(0, 1.42, 512)  # 1.42 seconds at 360 Hz
    clean_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)  # Simulated ECG
    noise = 0.3 * np.random.randn(512)  # Baseline wander noise
    noisy_ecg = clean_ecg + noise
    
    # Reshape for model input
    noisy_input = noisy_ecg.reshape(1, 512, 1).astype(np.float32)
    
    # Test best performing model
    best_model = 'Multibranch LANLD'
    if best_model in inference.get_available_models():
        print(f"\\nTesting {best_model}...")
        denoised = inference.denoise_ecg(noisy_input, best_model)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        time_ms = t * 1000  # Convert to milliseconds
        
        plt.subplot(3, 1, 1)
        plt.plot(time_ms, clean_ecg, 'g-', label='Original Clean ECG')
        plt.title('Original Clean ECG (Simulated)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(time_ms, noisy_ecg, 'r-', alpha=0.7, label='Noisy ECG')
        plt.title('Noisy ECG (Input)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        plt.plot(time_ms, denoised[0, :, 0], 'b-', label=f'{best_model} Output')
        plt.title('DeepFilter Denoised ECG')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_model_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Demo completed! Check 'multi_model_demo.png'")
    
    # Compare multiple models (if you want to test all)
    print("\\nTo compare all models, uncomment the following lines:")
    print("# results = inference.compare_models(noisy_input)")
    print("# This will run inference on all available models")

if __name__ == "__main__":
    demo()
'''
    
    with open(os.path.join(output_dir, 'inference_multi_model.py'), 'w', encoding='utf-8') as f:
        f.write(inference_script)
    
    # Create README
    readme_content = f'''# DeepFilter Model Deployment Package

This package contains all exported DeepFilter models for ECG denoising.

## Contents

- `models/`: Directory containing all exported SavedModel formats
- `model_registry.json`: Registry of available models with metadata
- `inference_multi_model.py`: Unified inference script for all models
- `README.md`: This file

## Available Models

{len([m for m in exported_models if m])} models exported:

'''
    
    for model_info in exported_models:
        if model_info:
            readme_content += f"- **{model_info['model_name']}**: {model_info['total_params']:,} parameters\\n"
    
    readme_content += f'''

## Model Performance (NV1 Results)

Based on test results, the models rank as follows (lower SSD = better):

1. **Multibranch LANLD**: Best overall performance
2. **FCN-DAE**: Good balance of speed and accuracy  
3. **Multibranch LANL**: Good performance
4. **Vanilla NL**: Moderate performance
5. **DRNN**: LSTM-based approach
6. **Vanilla L**: Simple linear CNN

## Technical Details

- **Input**: Noisy ECG signal (512 samples, 360 Hz, ~1.42 seconds)
- **Output**: Clean ECG signal (same dimensions)
- **Models**: TensorFlow SavedModel format
- **Deployment**: Ready for production use

## Requirements


'''
    
    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Deployment package created in '{output_dir}/'")
    print(f"   - {len([m for m in exported_models if m])} models exported")
    print(f"   - Unified inference script: inference_multi_model.py")
    print(f"   - Model registry: model_registry.json")
    print(f"   - Documentation: README.md")

def main():
    """Main export function"""
    
    print("=== DeepFilter Model Export - All Models ===")
    
    # Get all model configurations
    model_names = ['Vanilla L', 'Vanilla NL', 'FCN-DAE', 'DRNN', 'Multibranch LANL', 'Multibranch LANLD']
    
    # Check available models
    available_models = []
    for model_name in model_names:
        config = get_model_config(model_name)
        if config and os.path.exists(config['weight_file']):
            available_models.append((model_name, config))
            print(f"✅ Found: {model_name} ({config['weight_file']})")
        else:
            print(f"❌ Missing: {model_name}")
    
    if not available_models:
        print("❌ No trained models found!")
        return
    
    print(f"\n📦 Exporting {len(available_models)} models...")
    
    # Export all models
    exported_models = []
    successful_exports = 0
    
    for model_name, config in available_models:
        model_info = export_single_model(model_name, config)
        exported_models.append(model_info)
        
        if model_info:
            successful_exports += 1
            
            # Test the exported model
            test_result = test_exported_model(model_info)
            if test_result:
                model_info['test_result'] = test_result
    
    print(f"\n📊 EXPORT SUMMARY:")
    print(f"   Total models: {len(available_models)}")
    print(f"   Successfully exported: {successful_exports}")
    print(f"   Failed exports: {len(available_models) - successful_exports}")
    
    # Show model details
    print(f"\n📋 EXPORTED MODELS:")
    print(f"{'Model':<20} {'Parameters':<12} {'Status':<10} {'Test MSE':<12}")
    print("=" * 60)
    
    for model_info in exported_models:
        if model_info:
            test_mse = "N/A"
            if 'test_result' in model_info and model_info['test_result']:
                test_mse = f"{model_info['test_result']['mse_vs_ground_truth']:.6f}"
            
            print(f"{model_info['model_name']:<20} {model_info['total_params']:<12,} {'✅ OK':<10} {test_mse:<12}")
        else:
            print(f"{'Unknown':<20} {'N/A':<12} {'❌ Failed':<10} {'N/A':<12}")
    
    # Create deployment package
    if successful_exports > 0:
        create_deployment_package(exported_models)
        
        print(f"\n🎉 SUCCESS!")
        print(f"   - {successful_exports} models exported to 'exported_models/'")
        print(f"   - Deployment package created in 'deployment_package/'")
        print(f"   - Ready for production deployment!")
        
        # Show best model
        best_models = [m for m in exported_models if m and 'test_result' in m and m['test_result']]
        if best_models:
            best_model = min(best_models, key=lambda x: x['test_result']['mse_vs_ground_truth'])
            print(f"\n🏆 RECOMMENDED MODEL: {best_model['model_name']}")
            print(f"   Parameters: {best_model['total_params']:,}")
            print(f"   Test MSE: {best_model['test_result']['mse_vs_ground_truth']:.6f}")

if __name__ == "__main__":
    main()
