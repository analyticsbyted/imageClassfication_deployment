from preprocessing import preprocess_image
from tensorflow.keras.models import load_model
import numpy as np

def predict_image(image, model):
    # Preprocess the image
    image = preprocess_image(image)
    # Perform the prediction
    predictions = model.predict(image)
    # Get the predicted class label and confidence score
    predicted_class = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])
    return predicted_class, confidence_score
