"""
This script demonstrates how to use the colorization model on an example image.
It automatically selects a random grayscale image from the dataset and colorizes it.
"""

import os
import random
from colorize_personal_images import colorize_image, save_colorized_image
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

# Define custom objects for model loading
custom_objects = {'InstanceNormalization': InstanceNormalization}

def main():
    # Path to the model
    model_path = './results/gen0.h5'
    
    # Path to example grayscale images
    example_dir = './archive (3)/landscape Images/gray'
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you've extracted the model files from results.zip")
        return
    
    # Check if the example directory exists
    if not os.path.exists(example_dir):
        print(f"Error: Example image directory not found at {example_dir}")
        print("Please make sure you've extracted the archive (3).zip file")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            gen_model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get a list of grayscale images
    gray_images = [f for f in os.listdir(example_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not gray_images:
        print("No example images found.")
        return
    
    # Select a random image
    random_image = random.choice(gray_images)
    image_path = os.path.join(example_dir, random_image)
    
    print(f"Selected random example image: {random_image}")
    
    # Colorize and display the image
    print("Colorizing image...")
    colorized = colorize_image(gen_model, image_path)
    
    # Save the image
    if colorized is not None:
        output_path = save_colorized_image(gen_model, image_path)
        print(f"Test complete! Colorized image saved to: {output_path}")
        print("\nTo colorize your own images, use:")
        print("python colorize_personal_images.py --image path/to/your/image.jpg --save")

if __name__ == "__main__":
    main() 