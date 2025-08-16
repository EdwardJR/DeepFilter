#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Export All Models
Basic script to export all available DeepFilter models
"""

import os
import tensorflow as tf
from deepFilter.dl_models import *
from deepFilter.dl_pipeline import combined_ssd_mad_loss, ssd_loss, mad_loss

def main():
    print("=== Simple Export All Models ===")
    
    # Model configurations
    models = {
        'Vanilla_L': {
            'weights': 'Vanilla_L_weights.best.hdf5',
            'model_fn': lambda: deep_filter_vanilla_linear(signal_size=512),
            'loss': combined_ssd_mad_loss
        },
        'Vanilla_NL': {
            'weights': 'Vanilla_NL_weights.best.hdf5',
            'model_fn': lambda: deep_filter_vanilla_Nlinear(signal_size=512),
            'loss': combined_ssd_mad_loss
        },
        'FCN_DAE': {
            'weights': 'FCN_DAE_weights.best.hdf5',
            'model_fn': lambda: FCN_DAE(signal_size=512),
            'loss': ssd_loss
        },
        'DRNN': {
            'weights': 'DRNN_weights.best.hdf5',
            'model_fn': lambda: DRRN_denoising(signal_size=512),
            'loss': tf.keras.losses.mean_squared_error
        },
        'Multibranch_LANL': {
            'weights': 'Multibranch_LANL_weights.best.hdf5',
            'model_fn': lambda: deep_filter_I_LANL(signal_size=512),
            'loss': combined_ssd_mad_loss
        },
        'Multibranch_LANLD': {
            'weights': 'Multibranch_LANLD_weights.best.hdf5',
            'model_fn': lambda: deep_filter_model_I_LANL_dilated(signal_size=512),
            'loss': combined_ssd_mad_loss
        }
    }
    
    # Create output directory
    output_dir = 'exported_models'
    os.makedirs(output_dir, exist_ok=True)
    
    exported_count = 0
    
    for model_name, config in models.items():
        print(f"\n--- Processing {model_name} ---")
        
        if not os.path.exists(config['weights']):
            print(f"❌ Weights not found: {config['weights']}")
            continue
        
        try:
            # Create model
            model = config['model_fn']()
            
            # Compile model
            model.compile(
                loss=config['loss'],
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=['mse', 'mae']
            )
            
            # Load weights
            model.load_weights(config['weights'])
            print(f"✅ Loaded weights: {config['weights']}")
            
            # Export path
            export_path = os.path.join(output_dir, f"{model_name}_savedmodel")
            
            # Export to SavedModel
            model.save(export_path, save_format='tf')
            print(f"✅ Exported to: {export_path}")
            
            # Get model info
            params = model.count_params()
            print(f"📊 Parameters: {params:,}")
            
            exported_count += 1
            
        except Exception as e:
            print(f"❌ Error exporting {model_name}: {e}")
    
    print(f"\n🎉 Export Complete!")
    print(f"Successfully exported {exported_count}/{len(models)} models")
    print(f"Models saved in: {output_dir}/")

if __name__ == "__main__":
    main()
