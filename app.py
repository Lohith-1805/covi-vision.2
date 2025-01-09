from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__,
    static_url_path='/static',
    static_folder='static',
    template_folder='templates'
)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Define the model class
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

# Class mapping
class_mapping = {
    0: 'COVID',
    1: 'Normal',
    2: 'Pneumonia'
}

# Model path
MODEL_PATH = 'model_checkpoints/covid_model_converted.pth'

def load_model():
    try:
        print(f"Loading model from {MODEL_PATH}")
        model = ResNetClassifier(num_classes=3)
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        return None

# Initialize model
model = load_model()

if model is not None:
    print("Model initialized and ready for predictions")
    print(f"Model device: {next(model.parameters()).device}")
else:
    print("WARNING: Model failed to load - predictions will not be available")

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
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_pil = Image.fromarray(image)
        image_tensor = transform(image_pil).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities[0]).item()
        
        return jsonify({
            'success': True,
            'prediction': class_mapping[predicted_class]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)