from flask import Flask, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load the trained model
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
