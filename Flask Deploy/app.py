import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Load model with patch for 'batch_shape' error
import h5py
import json

try:
    model = tf.keras.models.load_model('waste_model.h5')
except ValueError as e:
    if "Unrecognized keyword arguments: ['batch_shape']" in str(e):
        print("Patching model config to remove 'batch_shape'...")
        with h5py.File('waste_model.h5', mode='r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError('No model config found in h5 file')
            model_config = json.loads(model_config)
            
            # Recursively rename 'batch_shape' to 'batch_input_shape' in InputLayer config
            def fix_input_layer_config(config):
                if isinstance(config, dict):
                    if 'class_name' in config and config['class_name'] == 'InputLayer':
                        if 'config' in config:
                            if 'batch_shape' in config['config'] and 'batch_input_shape' not in config['config']:
                                config['config']['batch_input_shape'] = config['config']['batch_shape']
                                del config['config']['batch_shape']
                            elif 'batch_shape' in config['config']:
                                del config['config']['batch_shape']
                    for key, value in config.items():
                        fix_input_layer_config(value)
                elif isinstance(config, list):
                    for item in config:
                        fix_input_layer_config(item)
            
            fix_input_layer_config(model_config)
            
            model = tf.keras.models.model_from_json(json.dumps(model_config))
            model.load_weights('waste_model.h5')
    else:
        raise e

# Load class names from the saved Labels.txt
with open('Labels.txt', 'r') as f:
    class_names = f.read().splitlines()

@app.route('/')
def home():
    return send_from_directory('./templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = img.resize((300, 300))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
