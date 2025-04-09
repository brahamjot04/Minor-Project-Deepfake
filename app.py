from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # Enables CORS for all domains on all routes

# Load the trained Keras model
model = tf.keras.models.load_model("face_classifier.h5")

# Define expected image size for model
img_size = (128, 128)

# Helper function to preprocess uploaded image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_bytes = image_file.read()
        img_array = preprocess_image(img_bytes)

        prediction = model.predict(img_array)[0][0]
        result = "Positive" if prediction > 0.5 else "Negative"
        confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

        return jsonify({
            'prediction': result,
            'confidence': round(confidence * 100, 2)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
