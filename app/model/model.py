from pathlib import Path
import re
from tensorflow.keras.models import load_model

# Get the base directory of the script
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Specify the path to the trained model
model_path = BASE_DIR / 'model'

# Load the trained model
model = load_model(model_path)

classes = []

def processing_image(image):
    """Process the image to be compatible with the model
    """
    # Convert the image to grayscale
    image

def processing_images(image):
    """Process the image to be compatible with the model
    """
    # Conve 
