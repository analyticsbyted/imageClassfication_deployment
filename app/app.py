from flask import Flask, render_template, request
from PIL import Image
from model.model import process_image

app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('app/templates/index.html')

# Define the route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    image_file = request.files['image']
    # Read the image file
    image = Image.open(image_file)
    # Process the image and make predictions
    predicted_class, confidence_score = process_image(image)
    # Render the result template with the predicted class and confidence score
    return render_template('app/templates/result.html', predicted_class=predicted_class, confidence_score=confidence_score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)
