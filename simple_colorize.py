"""
Simple script to colorize a single image with the pre-trained pix2pix GAN model.
Usage: python simple_colorize.py <path_to_image>
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

# Define custom objects for model loading
custom_objects = {'InstanceNormalization': InstanceNormalization}

def colorize_image(image_path, model_path='./results/gen0.h5'):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    # Load the model with custom objects scope
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model(model_path, compile=False)
    print("Model loaded successfully!")
    
    # Read and preprocess the image
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original for display
    original_img = img.copy()
    
    # Resize to 128x128 (the size expected by the model)
    img_resized = cv2.resize(img, (128, 128))
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_gray_3channel = np.stack([img_gray, img_gray, img_gray], axis=-1)
    
    # Normalize to [-1, 1] range
    img_gray_normalized = (img_gray_3channel.astype('float32') - 127.5) / 127.5
    
    # Add batch dimension
    img_gray_normalized = np.expand_dims(img_gray_normalized, axis=0)
    
    # Generate colorized image
    print("Colorizing the image...")
    colorized = model(img_gray_normalized, training=False)
    
    # Convert back from [-1, 1] to [0, 1] range
    colorized = (colorized[0] + 1) / 2.0
    gray_display = (img_gray_normalized[0] + 1) / 2.0
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image', fontsize=15)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gray_display, cmap='gray')
    plt.title('Grayscale Input', fontsize=15)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(colorized)
    plt.title('Colorized Output', fontsize=15)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the output
    output_dir = "colorized_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to RGB image (0-255)
    colorized_image = (colorized * 255).astype(np.uint8)
    
    # Save using OpenCV (convert back to BGR for saving)
    output_path = os.path.join(output_dir, f"colorized_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))
    
    print(f"Colorized image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_colorize.py <path_to_image>")
        print("Example: python simple_colorize.py ./archive (3)/landscape Images/gray/1.jpg")
    else:
        image_path = sys.argv[1]
        colorize_image(image_path) 