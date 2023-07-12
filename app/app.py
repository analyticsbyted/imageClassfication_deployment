from flask import Flask, render_template, request
from PIL import Image
from model.model import preprocess_image, predict_image, model, class_labels, convert_to_pil_image
from model.model import process_image

app = Flask(__name__, template_folder='templates', static_folder='static')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for image prediction

import base64
from io import BytesIO

# ...

def convert_to_base64(image):
    """Convert the PIL Image to base64-encoded string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_image

# ...

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    image_file = request.files['image']
    # Read the image file
    image = Image.open(image_file)
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Get the processed image as a PIL Image object
    processed_image_pil = convert_to_pil_image(processed_image)
    # Convert the processed image to base64
    processed_image_base64 = convert_to_base64(processed_image_pil)
    # Perform the prediction
    predicted_class, confidence_score = predict_image(processed_image, model, class_labels)
    # Render the result template with the prediction result and processed image
    return render_template('result.html', processed_image=processed_image_base64, predicted_class=predicted_class, confidence_score=confidence_score)

# Define the route for the thank you page
@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)
