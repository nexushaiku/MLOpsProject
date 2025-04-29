import io
import torch
import numpy as np
from PIL import Image
import mlflow.pytorch
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import psutil
# Define metrics
PREDICTION_COUNTER = Counter('prediction_requests_total', 'Total number of prediction requests', ['result'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')
DEFECT_RATIO = Gauge('defect_ratio', 'Ratio of defective prints detected')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

app = Flask(__name__)
# Track defect counts for ratio calculation
defect_count = 0
total_count = 0

# Load model
model = None
model_uri = None

def load_model(run_id):
    global model, model_uri
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model

def preprocess_image(image_bytes):
    # Convert to grayscale image
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
    image_array = (image_array - mean) / std[2]
    tensor = torch.tensor(image_array).unsqueeze(0).float()
    
    return tensor


@app.route('/predict', methods=['POST'])
def predict():
    global defect_count, total_count
    
    with PREDICTION_LATENCY.time():
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Preprocess the image
        tensor = preprocess_image(image_bytes)
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        # Update metrics after prediction
        total_count += 1
        if prediction == 0:  # Defective
            defect_count += 1
            PREDICTION_COUNTER.labels(result="defective").inc()
        else:
            PREDICTION_COUNTER.labels(result="non_defective").inc()
            
        DEFECT_RATIO.set(defect_count / total_count if total_count > 0 else 0)
        # Track resource usage
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        CPU_USAGE.set(psutil.cpu_percent())
        response = {
            'prediction': prediction,
            'confidence': float(torch.softmax(output, dim=1)[0, prediction].item())
        }
        
        return jsonify(response)
        
# Add metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start API server for handwriting recognition")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID for the model to serve")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()
    
    # Load the model
    load_model(args.run_id)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=args.port, debug=False)
