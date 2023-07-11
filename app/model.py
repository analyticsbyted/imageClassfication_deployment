from pathlib import Path
from tensorflow.keras.models import load_model
from app.utils.preprocessing import preprocess_image
from app.utils.prediction import predict_image
from PIL import Image


# Get the base directory of the script
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Specify the path to the trained model
model_path = BASE_DIR / 'model'

# Load the trained model
model = load_model(model_path)

# Define the class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def process_image(image):
    """Process the image to be compatible with the model
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Perform the prediction
    predicted_class, confidence_score = predict_image(preprocessed_image, model, class_labels)
    
    return predicted_class, confidence_score

def process_images(image_files):
    """Process a list of images to be compatible with the model
    """
    processed_images = []
    
    for image_file in image_files:
        # Read the image file
        image = Image.open(image_file)
        
        # Process the image
        predicted_class, confidence_score = process_image(image)
        
        processed_images.append({
            'image': image,
            'predicted_class': predicted_class,
            'confidence_score': confidence_score
        })
    
    return processed_images
