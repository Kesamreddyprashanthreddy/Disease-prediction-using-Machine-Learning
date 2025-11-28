import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image for model prediction.
    - Resizes image
    - Converts to NumPy array
    - Normalizes pixel values

    Args:
        image_path (str): Path to input image.
        target_size (tuple): Target size for model input.

    Returns:
        np.array: Preprocessed image array.
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array
