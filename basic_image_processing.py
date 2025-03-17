"""
Basic image processing script to test loading and processing images without requiring TensorFlow.
This helps test the preprocessing part of the colorization pipeline.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path, output_dir="processed_images"):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Read the image
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original for display
    original_img = img_rgb.copy()
    
    # Resize to 128x128 (the size expected by the model)
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Convert back to 3-channel grayscale
    img_gray_3channel = np.stack([img_gray, img_gray, img_gray], axis=-1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed grayscale image
    output_path = os.path.join(output_dir, f"gray_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, cv2.cvtColor(img_gray_3channel, cv2.COLOR_RGB2BGR))
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image', fontsize=15)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_gray_3channel)
    plt.title('Grayscale (Model Input)', fontsize=15)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_{os.path.basename(image_path)}"))
    plt.show()
    
    print(f"Image processing complete! Output saved to: {output_path}")
    print(f"Comparison image saved to: {os.path.join(output_dir, f'comparison_{os.path.basename(image_path)}')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python basic_image_processing.py <path_to_image>")
        print("Example: python basic_image_processing.py ./archive (3)/landscape Images/gray/999.jpg")
    else:
        image_path = sys.argv[1]
        process_image(image_path) 