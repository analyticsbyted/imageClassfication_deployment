from pathlib import Path
from tensorflow.keras.models import load_model

# Get the base directory of the script
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Specify the path to the trained model
model_path = BASE_DIR / 'model.h5'

# Load the trained model
model = load_model(model_path)


