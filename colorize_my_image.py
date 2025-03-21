import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

# Define custom objects for model loading
custom_objects = {'InstanceNormalization': InstanceNormalization}

def colorize_image(image_path):
    # Check if the input image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please update the image path in the script to point to your image.")
        return
    
    # Path to the model
    model_path = './results/gen0.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Read and process the image
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    
    # Resize the image to 128x128 (the expected model size)
    img_resized = cv2.resize(img, (128, 128))
    
    # Convert the resized image to grayscale and then stack to create a 3-channel image
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_gray_3channel = np.stack([img_gray, img_gray, img_gray], axis=-1)
    
    # Normalize the image to the [-1, 1] range
    img_gray_normalized = (img_gray_3channel.astype('float32') - 127.5) / 127.5
    
    # Add a batch dimension
    img_gray_normalized = np.expand_dims(img_gray_normalized, axis=0)
    
    # Use the model to colorize the image
    print("Colorizing the image...")
    colorized = model(img_gray_normalized, training=False)
    
    # Convert the output back from [-1, 1] to [0, 1]
    colorized = (colorized[0] + 1) / 2.0
    gray_display = (img_gray_normalized[0] + 1) / 2.0
    
    # Display the original, grayscale, and colorized images
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
    
    # Save the colorized output
    output_dir = "colorized_outputs"
    os.makedirs(output_dir, exist_ok=True)
    colorized_image = (colorized * 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"colorized_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))
    
    print(f"Colorized image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize an image using a pretrained model.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()
    
    colorize_image(args.image_path)
