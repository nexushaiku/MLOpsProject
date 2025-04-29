import torch
import mlflow
import mlflow.pytorch
import argparse
from data_utils import load_3D_dataset
from prometheus_client import Counter, Gauge, Summary
TEST_EVALUATIONS = Counter('test_evaluations_total', 'Total number of model evaluations')
TEST_ACCURACY = Gauge('test_accuracy', 'Test set accuracy')
INFERENCE_TIME = Summary('inference_time_seconds', 'Time taken for inference')

def evaluate_model(run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    TEST_EVALUATIONS.inc()
    @INFERENCE_TIME.time()
    def timed_inference(model, data):
        return model(data)
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    
    # Load test data
    _, _, test_loader = load_3D_dataset()
    
    # Evaluate
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = timed_inference(model, inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    TEST_ACCURACY.set(accuracy)
    print(f"Test accuracy for model {run_id}: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Printing model")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID for the model to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args.run_id)
