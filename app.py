from flask import Flask, render_template, request, jsonify, url_for
import torch
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__, 
    static_url_path='/static',  # explicitly set static url path
    static_folder='static',     # explicitly set static folder
    template_folder='templates' # explicitly set template folder
)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Load the PyTorch model
MODEL_PATH = 'model_checkpoints/covid_classifier_resnet_3classes.h5'

# Define the model class (same as in your notebook)
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

# Updated class mapping as requested
class_mapping = {
    0: 'COVID',      # Class 0 for COVID
    1: 'Normal',     # Class 1 for Normal
    2: 'Pneumonia'   # Class 2 (not 3) for Pneumonia
}

# Initialize model only if the file exists
model = None
try:
    model = ResNetClassifier(num_classes=3)
    if os.path.exists(MODEL_PATH):
        try:
            # First try with weights_only=True
            model.load_state_dict(
                torch.load(
                    MODEL_PATH, 
                    map_location=torch.device('cpu'),
                    weights_only=True
                ), 
                strict=False
            )
        except Exception as weight_error:
            print(f"Warning: Failed to load with weights_only=True: {weight_error}")
            # Fall back to regular loading if weights_only fails
            model.load_state_dict(
                torch.load(
                    MODEL_PATH, 
                    map_location=torch.device('cpu')
                ), 
                strict=False
            )
        model.eval()
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')

@app.route('/vaccinations')
def vaccinations():
    return render_template('vaccinations.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        })
        
    try:
        file = request.files['image']
        
        # Read and preprocess the image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))  # Match the IMG_HEIGHT and IMG_WIDTH
        
        # Apply the same normalization as in training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert to PIL Image for PyTorch transforms
        image_pil = Image.fromarray(image)
        image_tensor = transform(image_pil)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities[0]).item()
        
        result = {
            0: "COVID",
            1: "Normal",
            2: "Pneumonia"
        }
        
        return jsonify({
            'success': True,
            'prediction': result[predicted_class]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
