"""
Script to colorize preprocessed image data.
This requires TensorFlow and TensorFlow Addons to be properly installed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Check if required packages are installed
try:
    import tensorflow as tf
    import tensorflow_addons as tfa
    from tensorflow_addons.layers import InstanceNormalization
    from tensorflow.keras.models import load_model
except ImportError:
    print("ERROR: Required packages are not installed.")
    print("Please install the required packages with:")
    print("pip install tensorflow tensorflow-addons")
    print("Then run this script again.")
    exit(1)

def colorize_preprocessed(preprocessed_dir="preprocessed", model_path="./results/gen0.h5"):
    """Colorize preprocessed image data."""
    
    # Check if the preprocessed directory exists
    if not os.path.exists(preprocessed_dir):
        print(f"Error: Preprocessed directory not found at {preprocessed_dir}")
        print("Run 'python prepare_for_tf.py' first to create the preprocessed data.")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Find the input data file
    input_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.npy')]
    if not input_files:
        print(f"Error: No preprocessed input files found in {preprocessed_dir}")
        return
    
    input_file = os.path.join(preprocessed_dir, input_files[0])
    print(f"Found preprocessed input: {input_file}")
    
    try:
        # Load the preprocessed input
        print(f"Loading input data from {input_file}...")
        input_data = np.load(input_file)
        
        # Define custom objects for model loading
        custom_objects = {'InstanceNormalization': InstanceNormalization}
        
        # Load the model
        print(f"Loading model from {model_path}...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        # Generate colorized image
        print("Colorizing the image...")
        colorized = model(input_data, training=False)
        
        # Convert back from [-1, 1] to [0, 1] range
        colorized = (colorized[0] + 1) / 2.0
        gray_display = (input_data[0] + 1) / 2.0
        
        # Find the original image
        original_files = [f for f in os.listdir(preprocessed_dir) if f.startswith('original_')]
        if original_files:
            original_file = os.path.join(preprocessed_dir, original_files[0])
            original_img = cv2.imread(original_file)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        else:
            original_img = None
            
        # Display results
        if original_img is not None:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title('Original Image', fontsize=15)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(gray_display)
            plt.title('Grayscale Input', fontsize=15)
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(colorized)
            plt.title('Colorized Output', fontsize=15)
            plt.axis('off')
        else:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(gray_display)
            plt.title('Grayscale Input', fontsize=15)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(colorized)
            plt.title('Colorized Output', fontsize=15)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the colorized image
        output_dir = "colorized_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove the .npy extension and extract the original filename
        base_name = os.path.basename(input_file).replace("model_input_", "").replace(".npy", "")
        output_path = os.path.join(output_dir, f"colorized_{base_name}")
        
        # Convert to RGB image (0-255)
        colorized_img = (colorized * 255).astype(np.uint8)
        
        # Save using OpenCV (convert back to BGR for saving)
        cv2.imwrite(output_path, cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
        print(f"Colorized image saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error colorizing image: {str(e)}")
        return None

if __name__ == "__main__":
    print("Colorizing preprocessed image...")
    colorize_preprocessed() 