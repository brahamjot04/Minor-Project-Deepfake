from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
model = tf.keras.models.load_model("face_classifier.h5")
img_size = (128, 128)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('templates')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img_bytes = image_file.read()
    img_array = preprocess_image(img_bytes)

    prediction = model.predict(img_array)[0][0]
    result = "Positive" if prediction > 0.5 else "Negative"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

    return jsonify({'prediction': result, 'confidence': round(confidence * 100, 2)})

if __name__ == '__main__':
    app.run(debug=True)
