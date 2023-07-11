from PIL import Image
import numpy as np

def preprocess_image(image):
    # Convert the image to RGB mode if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Check the number of channels
    if len(image.split()) != 3:
        # Convert to RGB format if not already
        image = image.convert('RGB')

    # Resize the image to match the input shape of the model
    image = image.resize((32, 32))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image pixel values to the range of [0, 1]
    image = image.astype('float32') / 255.0
    # Expand the dimensions of the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    # Add the channel dimension
    image = np.expand_dims(image, axis=-1)
    return image

def preprocess_images(image_files):
    preprocessed_images = []
    for image_file in image_files:
        image = Image.open(image_file)
        preprocessed_image = preprocess_image(image)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images
