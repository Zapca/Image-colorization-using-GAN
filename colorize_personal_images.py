import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import os
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

# Define the custom objects dictionary for loading the model
custom_objects = {'InstanceNormalization': InstanceNormalization}

def preprocess_personal_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions for later display
    original_img = img.copy()
    
    # Resize to 128x128 (or the size expected by the model)
    img_resized = cv2.resize(img, (128, 128))
    
    # Convert to grayscale for the input
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_gray_3channel = np.stack([img_gray, img_gray, img_gray], axis=-1)
    
    # Normalize to [-1, 1] range as expected by the model
    img_gray_normalized = (img_gray_3channel.astype('float32') - 127.5) / 127.5
    
    return original_img, img_resized, img_gray_normalized

def colorize_image(gen_model, image_path):
    try:
        original_img, resized_img, gray_input = preprocess_personal_image(image_path)
        
        # Add batch dimension
        gray_input = np.expand_dims(gray_input, axis=0)
        
        # Generate the colorized image
        colorized = gen_model(gray_input, training=False)
        
        # Convert back from [-1, 1] to [0, 1] range
        colorized = (colorized[0] + 1) / 2.0
        gray_display = (gray_input[0] + 1) / 2.0
        
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
        
        return colorized
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def save_colorized_image(gen_model, image_path, output_directory="colorized_outputs"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # Process the image
        original_img, resized_img, gray_input = preprocess_personal_image(image_path)
        gray_input = np.expand_dims(gray_input, axis=0)
        colorized = gen_model(gray_input, training=False)
        colorized = (colorized[0] + 1) / 2.0
        
        # Convert to PIL Image and save
        colorized_image = Image.fromarray((colorized * 255).astype(np.uint8))
        base_filename = os.path.basename(image_path)
        output_path = os.path.join(output_directory, f"colorized_{base_filename}")
        colorized_image.save(output_path)
        
        print(f"Saved colorized image to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving colorized version of {image_path}: {str(e)}")
        return None

def process_image_directory(gen_model, directory_path, output_dir=None):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    # Get all image files in the directory
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory_path, file))
    
    print(f"Found {len(image_files)} images in directory")
    
    # Process each image
    for img_path in image_files:
        print(f"Processing: {os.path.basename(img_path)}")
        colorized = colorize_image(gen_model, img_path)
        
        if output_dir and colorized is not None:
            save_colorized_image(gen_model, img_path, output_dir)
    
    print("All images processed!")

def main():
    parser = argparse.ArgumentParser(description="Colorize personal images using the trained pix2pix GAN model")
    
    parser.add_argument('--model_path', type=str, default='./results/gen0.h5',
                        help='Path to the trained generator model (.h5 file)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to a single image to colorize')
    group.add_argument('--directory', type=str, help='Path to a directory of images to colorize')
    
    parser.add_argument('--save', action='store_true', help='Save the colorized images')
    parser.add_argument('--output_dir', type=str, default='colorized_outputs', 
                        help='Directory to save colorized images to (default: colorized_outputs)')
    
    args = parser.parse_args()
    
    # Load the generator model with custom objects
    print(f"Loading model from {args.model_path}...")
    with tf.keras.utils.custom_object_scope(custom_objects):
        gen_model = load_model(args.model_path, compile=False)
    print("Generator model loaded successfully!")
    
    if args.image:
        # Process a single image
        if args.save:
            save_colorized_image(gen_model, args.image, args.output_dir)
        else:
            colorize_image(gen_model, args.image)
    elif args.directory:
        # Process a directory of images
        process_image_directory(gen_model, args.directory, 
                               args.output_dir if args.save else None)

if __name__ == "__main__":
    main() 