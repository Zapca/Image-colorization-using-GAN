# How to Run the Pix2Pix GAN Colorization Model

This guide explains how to use the pre-trained GAN model to colorize your personal images.

## IMPORTANT: Model Loading Fix

The pix2pix model uses a custom layer called `InstanceNormalization` from the TensorFlow Addons package. To load the model properly, you must use the following approach:

```python
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import load_model

# Define custom objects for model loading
custom_objects = {'InstanceNormalization': InstanceNormalization}

# Load the model with custom object scope
with tf.keras.utils.custom_object_scope(custom_objects):
    model = load_model('./results/gen0.h5', compile=False)
```

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install tensorflow tensorflow-addons opencv-python numpy matplotlib pillow
```

## Option 1: Preprocessed Image Approach (Easiest)

We've created a preprocessing script that prepares your image without requiring TensorFlow:

1. Run the preprocessing script on your image:
   ```bash
   python prepare_for_tf.py
   ```

2. This will:
   - Create a `preprocessed` folder with your processed image
   - Save a README.txt with detailed instructions
   - Prepare everything needed for colorization

3. When you have access to TensorFlow, follow the instructions in the README.txt file.

## Option 2: Simple Script (Recommended)

Use the simple script once you have all dependencies installed:

```bash
python simple_colorize.py path/to/your/image.jpg
```

For example, to colorize one of the example grayscale images:

```bash
python simple_colorize.py "archive (3)/landscape Images/gray/999.jpg"
```

## Option 3: Full-Featured Script

For more options and batch processing:

```bash
# Colorize a single image
python colorize_personal_images.py --image path/to/your/image.jpg

# Colorize and save a single image
python colorize_personal_images.py --image path/to/your/image.jpg --save

# Colorize multiple images from a directory
python colorize_personal_images.py --directory path/to/your/images/folder --save
```

## Option 4: Try the Example Images

Test the model on the included example images:

```bash
# Create a directory for test outputs
mkdir test_outputs

# Run the script on a batch of example images
python colorize_personal_images.py --directory "archive (3)/landscape Images/gray" --save --output_dir test_outputs
```

## Tips for Best Results

1. **Image Size**: The model works best on landscape/outdoor images.
2. **Quality**: The model was trained on a specific dataset, so results may vary on different types of images.
3. **Processing**: If you're not satisfied with the results, try preprocessing your images:
   - Adjust brightness/contrast of your input image
   - Resize larger images before processing

## Troubleshooting

### "Unknown layer: 'Addons>InstanceNormalization'" Error

If you see this error, it means the model is using a custom layer and you need to use the special loading code shown at the top of this document.

### TensorFlow Installation Issues

TensorFlow can be challenging to install in some environments. If you're having trouble:

1. Try using a virtual environment:
   ```bash
   python -m venv tfenv
   # On Windows
   tfenv\Scripts\activate
   # On Linux/Mac
   source tfenv/bin/activate
   pip install tensorflow tensorflow-addons
   ```

2. Consider using the anaconda distribution with TensorFlow:
   ```bash
   conda create -n tf tensorflow
   conda activate tf
   pip install tensorflow-addons
   ```

3. Use the preprocessed approach (Option 1) and run the final colorization step in Google Colab or another environment with TensorFlow pre-installed.

### Model Performance

The model was trained on landscape images, so it works best on similar types of images. Results may vary with different types of images.

Enjoy colorizing your images! 