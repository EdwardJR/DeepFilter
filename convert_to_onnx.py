#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert All DeepFilter Models to ONNX
Convert all exported SavedModel formats to ONNX for maximum deployment flexibility
"""

import os
import numpy as np
import tensorflow as tf
from deepFilter.dl_pipeline import combined_ssd_mad_loss, ssd_loss, mad_loss

def install_tf2onnx():
    """Install tf2onnx if not available"""
    try:
        import tf2onnx
        print("✅ tf2onnx already installed")
        return True
    except ImportError:
        print("📦 Installing tf2onnx...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx", "onnx", "onnxruntime"])
            print("✅ tf2onnx installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install tf2onnx: {e}")
            print("Please install manually: pip install tf2onnx onnx onnxruntime")
            return False

def convert_savedmodel_to_onnx(savedmodel_path, onnx_path, model_name):
    """Convert a SavedModel to ONNX format"""
    
    print(f"\n--- Converting {model_name} ---")
    
    if not os.path.exists(savedmodel_path):
        print(f"❌ SavedModel not found: {savedmodel_path}")
        return False
    
    try:
        import tf2onnx
        import onnx
        
        # Method 1: Direct conversion from SavedModel
        try:
            print("🔄 Converting SavedModel to ONNX...")
            
            # Convert with tf2onnx
            model_proto, _ = tf2onnx.convert.from_saved_model(
                savedmodel_path,
                output_path=onnx_path,
                opset=13,  # Use ONNX opset 13 for good compatibility
                inputs_as_nchw=None
            )
            
            print(f"✅ Converted to: {onnx_path}")
            
            # Verify the ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model verification passed")
            
            # Get model info
            input_shape = None
            output_shape = None
            
            for input_info in onnx_model.graph.input:
                if input_info.name:
                    dims = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
                    input_shape = dims
                    break
            
            for output_info in onnx_model.graph.output:
                if output_info.name:
                    dims = [d.dim_value for d in output_info.type.tensor_type.shape.dim]
                    output_shape = dims
                    break
            
            print(f"📊 Input shape: {input_shape}")
            print(f"📊 Output shape: {output_shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Direct conversion failed: {e}")
            
            # Method 2: Load and convert via Keras model
            print("🔄 Trying alternative conversion method...")
            
            try:
                # Load the SavedModel
                model = tf.keras.models.load_model(savedmodel_path, custom_objects={
                    'combined_ssd_mad_loss': combined_ssd_mad_loss,
                    'ssd_loss': ssd_loss,
                    'mad_loss': mad_loss
                })
                
                # Create input signature
                input_signature = [tf.TensorSpec(shape=(None, 512, 1), dtype=tf.float32, name="input")]
                
                # Convert via Keras model
                model_proto, _ = tf2onnx.convert.from_keras(
                    model,
                    input_signature=input_signature,
                    opset=13,
                    output_path=onnx_path
                )
                
                print(f"✅ Alternative conversion successful: {onnx_path}")
                
                # Verify
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("✅ ONNX model verification passed")
                
                return True
                
            except Exception as e2:
                print(f"❌ Alternative conversion also failed: {e2}")
                return False
    
    except ImportError:
        print("❌ tf2onnx not available. Please install: pip install tf2onnx onnx onnxruntime")
        return False

def test_onnx_model(onnx_path, model_name):
    """Test the converted ONNX model"""
    
    try:
        import onnxruntime as ort
        
        print(f"🧪 Testing ONNX model: {model_name}")
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        print(f"   Input: {input_name} {input_shape}")
        print(f"   Output: {output_name} {output_shape}")
        
        # Create dummy input
        # Handle dynamic batch size
        test_shape = [1 if dim is None or dim == 'None' or (isinstance(dim, str) and 'unk' in dim.lower()) else dim for dim in input_shape]
        dummy_input = np.random.randn(*test_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        
        print(f"   ✅ Inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def create_onnx_inference_script(onnx_models, output_file='inference_onnx.py'):
    """Create a unified ONNX inference script"""
    
    script_content = f'''#!/usr/bin/env python3
"""
DeepFilter ONNX Inference Script
Unified inference for all ONNX-converted DeepFilter models
"""

import numpy as np
import onnxruntime as ort
import os
import matplotlib.pyplot as plt

class DeepFilterONNX:
    """ONNX inference class for DeepFilter models"""
    
    def __init__(self, models_dir='onnx_models'):
        self.models_dir = models_dir
        self.sessions = {{}}
        self.available_models = {list(onnx_models.keys())}
        
    def load_model(self, model_name):
        """Load a specific ONNX model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {{model_name}} not available. Available: {{self.available_models}}")
        
        if model_name not in self.sessions:
            model_path = os.path.join(self.models_dir, f"{{model_name}}.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {{model_path}}")
            
            self.sessions[model_name] = ort.InferenceSession(model_path)
            print(f"✅ Loaded {{model_name}}")
        
        return self.sessions[model_name]
    
    def denoise_ecg(self, noisy_ecg, model_name='Multibranch_LANLD'):
        """
        Denoise ECG using specified ONNX model
        
        Args:
            noisy_ecg: Input ECG signal (shape: [batch_size, 512, 1] or [512, 1])
            model_name: Name of the model to use
        
        Returns:
            clean_ecg: Denoised ECG signal
        """
        
        # Load model
        session = self.load_model(model_name)
        
        # Ensure correct input shape
        if len(noisy_ecg.shape) == 2:
            noisy_ecg = np.expand_dims(noisy_ecg, axis=0)
        
        # Convert to float32
        noisy_ecg = noisy_ecg.astype(np.float32)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        clean_ecg = session.run([output_name], {{input_name: noisy_ecg}})[0]
        
        return clean_ecg
    
    def compare_models(self, noisy_ecg, models_to_compare=None):
        """Compare multiple ONNX models on the same input"""
        
        if models_to_compare is None:
            models_to_compare = self.available_models
        
        results = {{}}
        
        for model_name in models_to_compare:
            try:
                clean_ecg = self.denoise_ecg(noisy_ecg, model_name)
                results[model_name] = clean_ecg
                print(f"✅ {{model_name}}: Success")
            except Exception as e:
                print(f"❌ {{model_name}}: {{e}}")
        
        return results
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        session = self.load_model(model_name)
        
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        return {{
            'input_name': input_info.name,
            'input_shape': input_info.shape,
            'input_type': input_info.type,
            'output_name': output_info.name,
            'output_shape': output_info.shape,
            'output_type': output_info.type
        }}

def demo():
    """Demo function"""
    
    print("=== DeepFilter ONNX Demo ===")
    
    # Initialize inference
    inference = DeepFilterONNX()
    
    print(f"Available models: {{inference.available_models}}")
    
    # Create dummy ECG data
    print("\\nCreating dummy noisy ECG data...")
    np.random.seed(42)
    
    # Simulate ECG-like signal with noise
    t = np.linspace(0, 1.42, 512)  # 1.42 seconds at 360 Hz
    clean_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    noise = 0.3 * np.random.randn(512)
    noisy_ecg = clean_ecg + noise
    
    # Reshape for model input
    noisy_input = noisy_ecg.reshape(1, 512, 1).astype(np.float32)
    
    # Test best model
    best_model = 'Multibranch_LANLD'
    if best_model in inference.available_models:
        print(f"\\nTesting {{best_model}}...")
        
        try:
            # Get model info
            info = inference.get_model_info(best_model)
            print(f"Model info: {{info}}")
            
            # Run inference
            denoised = inference.denoise_ecg(noisy_input, best_model)
            
            print(f"✅ Inference successful!")
            print(f"Input shape: {{noisy_input.shape}}")
            print(f"Output shape: {{denoised.shape}}")
            
            # Plot results
            plt.figure(figsize=(12, 8))
            
            time_ms = t * 1000
            
            plt.subplot(3, 1, 1)
            plt.plot(time_ms, clean_ecg, 'g-', label='Original Clean')
            plt.title('Original Clean ECG (Simulated)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 2)
            plt.plot(time_ms, noisy_ecg, 'r-', alpha=0.7, label='Noisy Input')
            plt.title('Noisy ECG (Input)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 3)
            plt.plot(time_ms, denoised[0, :, 0], 'b-', label=f'{{best_model}} ONNX Output')
            plt.title('ONNX Model Denoised ECG')
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('onnx_demo.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("✅ Demo completed! Check 'onnx_demo.png'")
            
        except Exception as e:
            print(f"❌ Demo failed: {{e}}")
    
    # Show available models and their info
    print("\\n=== Model Information ===")
    for model_name in inference.available_models:
        try:
            info = inference.get_model_info(model_name)
            print(f"{{model_name}}:")
            print(f"  Input: {{info['input_shape']}} ({{info['input_type']}})")
            print(f"  Output: {{info['output_shape']}} ({{info['output_type']}})")
        except Exception as e:
            print(f"{{model_name}}: Error - {{e}}")

if __name__ == "__main__":
    demo()
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ ONNX inference script created: {output_file}")

def main():
    """Main conversion function"""
    
    print("=== DeepFilter SavedModel to ONNX Conversion ===")
    
    # Check if tf2onnx is available
    if not install_tf2onnx():
        print("❌ Cannot proceed without tf2onnx")
        return
    
    # Define model mappings
    savedmodel_dir = 'exported_models'
    onnx_dir = 'onnx_models'
    
    # Create ONNX output directory
    os.makedirs(onnx_dir, exist_ok=True)
    
    # Model mappings (SavedModel folder -> ONNX file)
    model_mappings = {
        'Vanilla_L': 'Vanilla_L_savedmodel',
        'Vanilla_NL': 'Vanilla_NL_savedmodel', 
        'FCN_DAE': 'FCN_DAE_savedmodel',
        'DRNN': 'DRNN_savedmodel',
        'Multibranch_LANL': 'Multibranch_LANL_savedmodel',
        'Multibranch_LANLD': 'Multibranch_LANLD_savedmodel'
    }
    
    # Check available SavedModels
    available_models = {}
    for model_name, savedmodel_folder in model_mappings.items():
        savedmodel_path = os.path.join(savedmodel_dir, savedmodel_folder)
        if os.path.exists(savedmodel_path):
            available_models[model_name] = savedmodel_path
            print(f"✅ Found SavedModel: {model_name}")
        else:
            print(f"❌ Missing SavedModel: {model_name}")
    
    if not available_models:
        print("❌ No SavedModels found! Please run simple_export_all.py first.")
        return
    
    print(f"\\n🔄 Converting {len(available_models)} models to ONNX...")
    
    # Convert each model
    successful_conversions = {}
    failed_conversions = []
    
    for model_name, savedmodel_path in available_models.items():
        onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
        
        success = convert_savedmodel_to_onnx(savedmodel_path, onnx_path, model_name)
        
        if success:
            successful_conversions[model_name] = onnx_path
            
            # Test the ONNX model
            test_success = test_onnx_model(onnx_path, model_name)
            if not test_success:
                print(f"⚠️  {model_name} converted but failed testing")
        else:
            failed_conversions.append(model_name)
    
    # Summary
    print(f"\\n📊 CONVERSION SUMMARY:")
    print(f"   Total models: {len(available_models)}")
    print(f"   Successfully converted: {len(successful_conversions)}")
    print(f"   Failed conversions: {len(failed_conversions)}")
    
    if successful_conversions:
        print(f"\\n✅ SUCCESSFULLY CONVERTED:")
        for model_name, onnx_path in successful_conversions.items():
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            print(f"   - {model_name}: {onnx_path} ({file_size:.1f} MB)")
    
    if failed_conversions:
        print(f"\\n❌ FAILED CONVERSIONS:")
        for model_name in failed_conversions:
            print(f"   - {model_name}")
    
    # Create inference script
    if successful_conversions:
        create_onnx_inference_script(successful_conversions)
        
        # Create README for ONNX models
        readme_content = f'''# DeepFilter ONNX Models

This directory contains ONNX-converted DeepFilter models for maximum deployment flexibility.

## Available Models

{len(successful_conversions)} models successfully converted:

'''
        
        for model_name in successful_conversions.keys():
            readme_content += f"- `{model_name}.onnx`\\n"
        
        readme_content += f'''
## Usage

### Python with ONNX Runtime
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('{list(successful_conversions.keys())[0]}.onnx')

# Prepare input (noisy ECG)
noisy_ecg = np.random.randn(1, 512, 1).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
clean_ecg = session.run([output_name], {{input_name: noisy_ecg}})[0]
```

### Run Demo
```bash
python inference_onnx.py
```

## Model Specifications

- **Input**: (batch_size, 512, 1) - Noisy ECG signal
- **Output**: (batch_size, 512, 1) - Clean ECG signal  
- **Sampling Rate**: 360 Hz
- **Signal Duration**: ~1.42 seconds
- **Format**: ONNX (Open Neural Network Exchange)

## Deployment Advantages

- **Cross-platform**: Run on any ONNX-compatible runtime
- **Language agnostic**: Use from Python, C++, C#, Java, JavaScript, etc.
- **Hardware optimized**: Automatic optimization for target hardware
- **Production ready**: Industry standard format for ML deployment

## Requirements

```bash
pip install onnxruntime numpy matplotlib
```

For GPU acceleration:
```bash
pip install onnxruntime-gpu
```
'''
        
        with open(os.path.join(onnx_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"\\n🎉 SUCCESS!")
        print(f"   - {len(successful_conversions)} models converted to ONNX")
        print(f"   - Models saved in: {onnx_dir}/")
        print(f"   - Inference script: inference_onnx.py")
        print(f"   - Documentation: {onnx_dir}/README.md")
        print(f"   - Ready for cross-platform deployment!")
        
        # Show recommended model
        if 'Multibranch_LANLD' in successful_conversions:
            print(f"\\n🏆 RECOMMENDED: Multibranch_LANLD.onnx (best performance)")
        elif successful_conversions:
            first_model = list(successful_conversions.keys())[0]
            print(f"\\n📋 Available: {first_model}.onnx")

if __name__ == "__main__":
    main()
