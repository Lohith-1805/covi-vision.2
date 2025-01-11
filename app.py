import logging
import os
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static',
    template_folder='templates'
)

# Update model path to be relative to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_checkpoints', 'covid_classifier_resnet_3classes.h5')

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(weights=None)
        
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

class_mapping = {
    0: 'COVID',
    1: 'Normal',
    2: 'Pneumonia'
}

model = None

def load_model():
    global model
    try:
        if model is None:
            logger.info(f"Loading model from path: {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
            model = ResNetClassifier(num_classes=3)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file was actually sent
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400

        # Load model
        try:
            load_model()
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Model loading failed: {str(e)}'
            }), 500

        # Process image
        try:
            image_bytes = file.read()
            image = cv2.imdecode(np.fromstring(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            
            image = cv2.resize(image, (224, 224))
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image_pil = Image.fromarray(image)
            image_tensor = transform(image_pil)
            image_tensor = image_tensor.unsqueeze(0)

        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Image processing failed: {str(e)}'
            }), 400

        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities[0]).item()
                confidence = float(probabilities[0][predicted_class])

            return jsonify({
                'success': True,
                'prediction': class_mapping[predicted_class],
                'confidence': f"{confidence * 100:.2f}%"
            })

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"General error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)