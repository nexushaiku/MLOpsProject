import io
import os
import torch
import numpy as np
from PIL import Image
import mlflow.pytorch
from flask import Flask, request, jsonify
import argparse
import pandas as pd
import datetime
import threading
import time

# --- Prometheus Metrics Setup ---
from prometheus_client import Counter, Histogram, Gauge, make_wsgi_app, REGISTRY
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import psutil # For resource monitoring

# --- Define or Retrieve Metrics ---
# Use try-except ValueError to handle potential duplicate registration
try:
    PREDICTION_COUNTER = Counter(
        'prediction_requests_total',
        'Total number of prediction requests',
        ['result'] # Label to distinguish defective/non-defective
    )
except ValueError:
    print("Metric 'prediction_requests_total' already exists. Retrieving.")
    # Retrieve existing metric object from the registry (using internal access carefully)
    PREDICTION_COUNTER = REGISTRY._metrics.get('prediction_requests_total')

try:
    PREDICTION_LATENCY = Histogram(
        'prediction_latency_seconds',
        'Time spent processing prediction request'
    )
except ValueError:
    print("Metric 'prediction_latency_seconds' already exists. Retrieving.")
    PREDICTION_LATENCY = REGISTRY._metrics.get('prediction_latency_seconds')

try:
    DEFECT_RATIO = Gauge(
        'defect_ratio',
        'Ratio of defective prints detected (prediction=0)'
    )
except ValueError:
    print("Metric 'defect_ratio' already exists. Retrieving.")
    DEFECT_RATIO = REGISTRY._metrics.get('defect_ratio')

try:
    MEMORY_USAGE = Gauge(
        'api_memory_usage_bytes',
        'Memory usage of the API process in bytes'
    )
except ValueError:
    print("Metric 'api_memory_usage_bytes' already exists. Retrieving.")
    MEMORY_USAGE = REGISTRY._metrics.get('api_memory_usage_bytes')

try:
    CPU_USAGE = Gauge(
        'api_cpu_usage_percent',
        'CPU usage percentage of the API process'
    )
except ValueError:
    print("Metric 'api_cpu_usage_percent' already exists. Retrieving.")
    CPU_USAGE = REGISTRY._metrics.get('api_cpu_usage_percent')

# --- Global Variables ---
app = Flask(__name__)
model = None
model_uri = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
defect_count = 0
total_count = 0
LOG_FILE = "prediction_logs.csv" # Log file for drift detection input
log_lock = threading.Lock()

# --- Model Loading ---
def load_model_from_mlflow(run_id):
    global model, model_uri, device
    try:
        model_uri = f"runs:/{run_id}/model"
        print(f"Loading model from: {model_uri}")
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device) # Move model to the correct device
        model.eval()     # Set model to evaluation mode
        print(f"Model loaded successfully and moved to {device}.")
        return model
    except Exception as e:
        print(f"FATAL: Failed to load model from {model_uri}. Error: {e}")
        model = None # Ensure model is None if loading failed
        return None

# --- Image Preprocessing (Ensure consistency with training) ---
def preprocess_image(image_bytes):
    """Preprocesses image bytes. MUST match training preprocessing."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((128, 128))
        image_array = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = image_array.transpose((2, 0, 1)) # HWC to CHW
        normalized_array = (image_array - mean[:, None, None]) / std[:, None, None]
        tensor = torch.tensor(normalized_array, dtype=torch.float32)
        return tensor
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Prediction Logging for Drift ---
def log_prediction_data(input_tensor, prediction, confidence):
    """Logs prediction data (including image stats) asynchronously."""
    global log_lock
    try:
        timestamp = datetime.datetime.now().isoformat()
        image_mean = input_tensor.mean().item()
        image_std = input_tensor.std().item()
        log_entry = {
            "timestamp": [timestamp], "image_mean": [image_mean], "image_std": [image_std],
            "prediction": [prediction], "confidence": [confidence]
        }
        df_new = pd.DataFrame(log_entry)
        with log_lock:
            header = not os.path.exists(LOG_FILE)
            df_new.to_csv(LOG_FILE, mode='a', header=header, index=False)
    except Exception as e:
        print(f"Error logging prediction data: {e}")

# --- API Endpoint (/predict) ---
@app.route('/predict', methods=['POST'])
def predict():
    global model, defect_count, total_count, device

    if model is None: return jsonify({'error': 'Model not loaded'}), 500

    with PREDICTION_LATENCY.time(): # Time the request
        if 'image' not in request.files: return jsonify({'error': 'No image file provided'}), 400

        try:
            image_file = request.files['image']
            image_bytes = image_file.read()

            input_tensor = preprocess_image(image_bytes)
            if input_tensor is None: return jsonify({'error': 'Image preprocessing failed'}), 400

            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                prediction = predicted_class.item()
                confidence_score = confidence.item() * 100

            # --- Update Prometheus Metrics ---
            total_count += 1
            label = "defective" if prediction == 0 else "non_defective"
            if prediction == 0: defect_count += 1

            # Check if metric objects exist before using them (in case retrieval failed)
            if PREDICTION_COUNTER: PREDICTION_COUNTER.labels(result=label).inc()
            if DEFECT_RATIO: DEFECT_RATIO.set(defect_count / total_count if total_count > 0 else 0)

            process = psutil.Process(os.getpid())
            if MEMORY_USAGE: MEMORY_USAGE.set(process.memory_info().rss)
            if CPU_USAGE: CPU_USAGE.set(process.cpu_percent(interval=None))

            # --- Log Prediction for Drift Detection (Asynchronously) ---
            log_thread = threading.Thread(target=log_prediction_data, args=(input_tensor, prediction, confidence_score))
            log_thread.start()

            response = {
                'prediction': prediction,
                'status': 'Defective' if prediction == 0 else 'Non-Defective',
                'confidence': round(confidence_score, 2)
            }
            return jsonify(response)

        except Exception as e:
            print(f"Error during prediction: {e}")
            # Consider logging the full traceback for debugging
            # import traceback
            # print(traceback.format_exc())
            return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

# --- Prometheus Metrics Endpoint Setup ---
# Wrap the app with middleware for the /metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app(REGISTRY) # Use the same registry
})

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start API server for 3D Printing Defect Detection")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID for the model to serve")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    args = parser.parse_args()
    print(f"API Server starting up...")
    print(f"Using device: {device}")

    # Load the model at startup
    load_model_from_mlflow(args.run_id)

    if model is None:
        print("Exiting: Model could not be loaded.")
        exit(1) # Exit if model loading failed

    # Check if metric objects were successfully defined/retrieved
    if not all([PREDICTION_COUNTER, PREDICTION_LATENCY, DEFECT_RATIO, MEMORY_USAGE, CPU_USAGE]):
         print("WARN: Some Prometheus metrics could not be initialized/retrieved.")

    print(f"Starting Flask API server on http://0.0.0.0:{args.port}")
    print(f"Prometheus metrics available at http://0.0.0.0:{args.port}/metrics")
    # Use a production WSGI server like waitress or gunicorn in production
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=args.port)
    app.run(host='0.0.0.0', port=args.port, debug=False)
