from flask import Flask, jsonify, request, render_template
import numpy as np
import onnxruntime as ort
from flask_cors import CORS
from PIL import Image

CLASS_NAME = ['Aluminium', 'Carton', 'E-waste', 'Glass', 'Organic_Waste', 'Paper_and_Cardboard', 'Plastics', 'Textiles', 'Wood']

app = Flask(__name__)
CORS(app)

# Build Model
model_path = './model4_resnet.onnx'
ort_session = ort.InferenceSession(model_path)

# Define a route for the GET request
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def preprocess_image(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    # Resize and Center Crop
    img_np = np.array(img.resize((224, 224)))  # Resize
    h, w = img_np.shape[:2]
    start_h = (h - 224) // 2
    start_w = (w - 224) // 2
    img_np = img_np[start_h:start_h + 224, start_w:start_w + 224]  # Center Crop

    # Convert to Tensor
    img_np = np.transpose(img_np, (2, 0, 1))  # Transpose to (C, H, W)
    img_tensor = img_np.astype(np.float32) / 255.0  # Convert to float32 and normalize to [0, 1]

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.299, 0.224, 0.225])
    img_normalized = (img_tensor - mean[:, None, None]) / std[:, None, None]

    img_normalized = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    return img_normalized

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image file
    image = request.files['image']

    input_image = preprocess_image(Image.open(image))
    input_image = input_image.astype(np.float32)

    input_name = ort_session.get_inputs()[0].name
    
    # Inferencing
    outputs = ort_session.run(None, {input_name: input_image})
    predictions = outputs[0]

    # Get class probabilities using softmax
    softmax_output = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    probabilities = softmax_output[0]

    # Get the index of the highest probability
    class_index = np.argmax(probabilities)

    # Get the predicted class and probability
    predicted_class = CLASS_NAME[class_index]
    probability = probabilities[class_index]

    # Sort class probabilities in descending order
    class_probs = list(zip(CLASS_NAME, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Return JSON response
    return jsonify({
        'predicted_class': predicted_class,
        'probability': float(probability),
        'class_probs': [(class_name, float(prob)) for class_name, prob in class_probs]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
