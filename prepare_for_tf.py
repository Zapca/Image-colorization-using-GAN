"""
Helper script to preprocess your personal image and save it as a NumPy array
that can be loaded later for colorization once TensorFlow is properly installed.
"""

import os
import cv2
import numpy as np

def preprocess_and_save_image(image_path, output_dir="preprocessed"):
    """Preprocess an image and save the input tensor as a NumPy array."""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False
    
    try:
        # Read the image
        print(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return False
        
        # Convert to RGB (from BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save the original image
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"original_{os.path.basename(image_path)}"), 
                   cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        # Resize to 128x128 (the size expected by the model)
        img_resized = cv2.resize(img_rgb, (128, 128))
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Convert back to 3-channel grayscale
        img_gray_3channel = np.stack([img_gray, img_gray, img_gray], axis=-1)
        
        # Save the preprocessed grayscale image
        cv2.imwrite(os.path.join(output_dir, f"gray_{os.path.basename(image_path)}"), 
                   cv2.cvtColor(img_gray_3channel, cv2.COLOR_RGB2BGR))
        
        # Normalize to [-1, 1] range (as expected by the model)
        img_gray_normalized = (img_gray_3channel.astype('float32') - 127.5) / 127.5
        
        # Add batch dimension
        model_input = np.expand_dims(img_gray_normalized, axis=0)
        
        # Save the preprocessed input as a numpy array
        np.save(os.path.join(output_dir, f"model_input_{os.path.basename(image_path)}.npy"), model_input)
        
        print(f"Image preprocessing complete!")
        print(f"Original image saved to: {os.path.join(output_dir, f'original_{os.path.basename(image_path)}')}")
        print(f"Grayscale image saved to: {os.path.join(output_dir, f'gray_{os.path.basename(image_path)}')}")
        print(f"Model input saved to: {os.path.join(output_dir, f'model_input_{os.path.basename(image_path)}.npy')}")
        
        # Create a README file with instructions for using the preprocessed data
        with open(os.path.join(output_dir, "README.txt"), "w") as f:
            f.write("### Preprocessed Image Data ###\n\n")
            f.write("This directory contains preprocessed image data ready for colorization.\n\n")
            f.write("To colorize this image with the pix2pix model, you need to:\n\n")
            f.write("1. Install TensorFlow and TensorFlow Addons:\n")
            f.write("   pip install tensorflow tensorflow-addons\n\n")
            f.write("2. Run the following Python code:\n\n")
            f.write("```python\n")
            f.write("import numpy as np\n")
            f.write("import tensorflow as tf\n")
            f.write("import tensorflow_addons as tfa\n")
            f.write("from tensorflow_addons.layers import InstanceNormalization\n")
            f.write("import matplotlib.pyplot as plt\n")
            f.write("from tensorflow.keras.models import load_model\n")
            f.write("import cv2\n")
            f.write("import os\n\n")
            f.write("# Define custom objects for model loading\n")
            f.write("custom_objects = {'InstanceNormalization': InstanceNormalization}\n\n")
            f.write(f"# Load the preprocessed input\n")
            f.write(f"input_data = np.load('{os.path.join(output_dir, f'model_input_{os.path.basename(image_path)}.npy')}')\n\n")
            f.write("# Load the model\n")
            f.write("with tf.keras.utils.custom_object_scope(custom_objects):\n")
            f.write("    model = load_model('./results/gen0.h5', compile=False)\n\n")
            f.write("# Generate colorized image\n")
            f.write("colorized = model(input_data, training=False)\n\n")
            f.write("# Convert back from [-1, 1] to [0, 1] range for display\n")
            f.write("colorized = (colorized[0] + 1) / 2.0\n\n")
            f.write("# Display the colorized image\n")
            f.write("plt.figure(figsize=(10, 10))\n")
            f.write("plt.imshow(colorized)\n")
            f.write("plt.axis('off')\n")
            f.write("plt.title('Colorized Image')\n")
            f.write("plt.show()\n\n")
            f.write("# Save the colorized image\n")
            f.write("colorized_img = (colorized * 255).astype(np.uint8)\n")
            f.write("cv2.imwrite('colorized_output.jpg', cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))\n")
            f.write("```\n\n")
            f.write("This will colorize your preprocessed image and save the result as 'colorized_output.jpg'.\n")
        
        print(f"README with instructions saved to: {os.path.join(output_dir, 'README.txt')}")
        return True
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return False

if __name__ == "__main__":
    # Use the image path from the notebook
    image_path = 'C://Users/Shreyas/Downloads/pexels-ulltangfilms-285286.jpg'
    
    # Check if the path exists or ask for a new one
    if not os.path.exists(image_path):
        print(f"The image at {image_path} was not found.")
        new_path = input("Please enter the path to your image: ")
        if new_path:
            image_path = new_path
    
    success = preprocess_and_save_image(image_path)
    
    if success:
        print("\nImage preprocessing completed successfully!")
        print("\nTo colorize this image:")
        print("1. Install TensorFlow with TensorFlow Addons in a compatible environment")
        print("2. Follow the instructions in the preprocessed/README.txt file")
    else:
        print("\nImage preprocessing failed. Please check the error messages above.") 