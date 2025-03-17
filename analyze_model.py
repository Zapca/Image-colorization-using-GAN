"""
Script to analyze the structure of the model without loading it fully.
This will help identify the exact requirements for loading.
"""

import os
import h5py

def analyze_h5_model(model_path):
    """Analyze the structure of an H5 model file."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Analyzing model at {model_path}...")
    
    try:
        with h5py.File(model_path, 'r') as f:
            # Print the top-level keys (groups)
            print("\nTop-level keys in the H5 file:")
            for key in f.keys():
                print(f"- {key}")
                
                # If this is the model_weights group, explore its structure
                if key == 'model_weights':
                    print("\nExploring model_weights group:")
                    weights_group = f[key]
                    for layer_name in weights_group.keys():
                        print(f"  Layer: {layer_name}")
                        layer_group = weights_group[layer_name]
                        
                        # Check for custom objects in layer names
                        if 'InstanceNormalization' in layer_name:
                            print(f"    *** Custom layer detected: InstanceNormalization in {layer_name} ***")
                        
                        # Print weights for this layer
                        if isinstance(layer_group, h5py.Group):
                            for weight_name in layer_group.keys():
                                print(f"    Weight: {weight_name}")
            
            # Check for attributes that might contain model config
            print("\nFile attributes:")
            for attr_name in f.attrs:
                attr_value = f.attrs[attr_name]
                print(f"- {attr_name}")
                
                # Try to decode if it's a string-like object
                if attr_name == 'model_config' and hasattr(attr_value, 'decode'):
                    try:
                        import json
                        config = json.loads(attr_value.decode('utf-8'))
                        print("\nModel configuration found:")
                        print(f"Model class name: {config.get('class_name', 'Unknown')}")
                        
                        # Check layers
                        if 'config' in config and 'layers' in config['config']:
                            custom_layers = []
                            for layer in config['config']['layers']:
                                layer_class = layer.get('class_name', 'Unknown')
                                if layer_class == 'InstanceNormalization' or 'InstanceNormalization' in layer_class:
                                    custom_layers.append(layer_class)
                            
                            if custom_layers:
                                print("\nCustom layers found in configuration:")
                                for layer in custom_layers:
                                    print(f"- {layer}")
                    except Exception as e:
                        print(f"  (Error decoding model_config: {str(e)})")
    
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")
    
    # Provide a solution regardless of analysis results
    print("\n\n===== SOLUTION =====")
    print("Based on previous error messages, this model uses InstanceNormalization from tensorflow_addons.")
    print("To load this model, you need to use the following code:")
    print("\n```python")
    print("import tensorflow as tf")
    print("import tensorflow_addons as tfa")
    print("from tensorflow_addons.layers import InstanceNormalization")
    print("from tensorflow.keras.models import load_model")
    print("\n# Register the custom layer")
    print("custom_objects = {'InstanceNormalization': InstanceNormalization}")
    print("\n# Load the model with custom object scope")
    print("with tf.keras.utils.custom_object_scope(custom_objects):")
    print("    model = load_model('./results/gen0.h5', compile=False)")
    print("```")

if __name__ == "__main__":
    model_path = './results/gen0.h5'
    analyze_h5_model(model_path) 