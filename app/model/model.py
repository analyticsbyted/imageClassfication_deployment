from pathlib import Path
from tensorflow.keras.models import load_model
from skimage import transform
from skimage.color import rgba2rgb
from skimage.io import imread
from PIL import Image
import numpy as np

# Get the base directory of the script
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

# Specify the path to the trained model
model_path = BASE_DIR / 'model'

# class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
model = load_model(str(model_path))

def preprocess_image(image):
    """Preprocess the image to make it compatible with the model"""
    if isinstance(image, np.ndarray):
        # Convert RGBA to RGB if necessary
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
        
        # Check the number of dimensions
        if len(image.shape) == 3:
            # Resize the image to match the input shape of the model
            image = transform.resize(image, (32, 32, image.shape[-1]))
            # Convert the image to a numpy array
            image = np.array(image)
            # Normalize the image pixel values to the range of [0, 1]
            image = image.astype('float32') / 255.0
            # Add the batch dimension
            image = np.expand_dims(image, axis=0)
        
    elif isinstance(image, Image.Image):
        # Convert the image to RGB mode if it has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Convert grayscale images to RGB
        if image.mode == 'L':
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
    
    else:
        raise ValueError("Invalid image format. Supported formats are numpy.ndarray and PIL.Image.Image.")
    
    return image


def convert_to_pil_image(image):
    """Convert the processed image to PIL Image object"""
    # Convert the image from NumPy array to PIL Image
    image = (image[0] * 255).astype(np.uint8)
    image = Image.fromarray(image)
    
    return image


def get_processed_image(image):
    """Get the processed image as a PIL Image object"""
    return convert_to_pil_image(image)


def predict_image(image, model, class_labels):
    """Perform the prediction on the preprocessed image using the loaded model"""
    # Preprocess the image
    image = preprocess_image(image)
    
    # Perform the prediction
    predictions = model.predict(image)
    
    # Get the predicted class label and confidence score
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence_score = np.max(predictions[0])
    
    return predicted_class, confidence_score


def process_image(image):
    """Process the image to be compatible with the model"""
    predicted_class, confidence_score = predict_image(image, model, class_labels)
    return predicted_class, confidence_score


def process_images(image_files):
    """Process a list of images to be compatible with the model"""
    processed_images = []
    
    for image_file in image_files:
        # Read the image file
        image = imread(image_file)
        
        # Process the image
        predicted_class, confidence_score = process_image(image)
        
        processed_images.append({
            'image': image,
            'predicted_class': predicted_class,
            'confidence_score': confidence_score
        })
    
    return processed_images
