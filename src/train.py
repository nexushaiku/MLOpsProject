import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import argparse
from torchinfo import summary
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

from data_utils import load_3D_dataset
from model import get_model

# Define metrics
TRAINING_ITERATIONS = Counter('training_iterations_total', 'Total number of training iterations')
TRAINING_LOSS = Gauge('training_loss', 'Current training loss value')
VALIDATION_ACCURACY = Gauge('validation_accuracy', 'Validation accuracy')
BATCH_DURATION = Histogram('batch_duration_seconds', 'Time for batch processing')

# Start metrics server
start_http_server(8000)

def train_model(data_dir, train_ratio, val_ratio, test_ratio, experiment_name="3D_Print_Defect_Detector"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    
    # Data loading with specified split ratios
    train_loader, val_loader, test_loader = load_3D_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Initialize model, loss, and optimizer
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer.__class__.__name__
        })
        
        # Log model summary as an artifact
        model_summary = summary(model, input_size=(batch_size, 3, 128, 128), verbose=0)
        with open("model_summary.txt", "w") as f:
            f.write(str(model_summary))
        mlflow.log_artifact("model_summary.txt")
        
        # Initialize lists to store metrics for plotting
        train_losses = []
        val_losses = []
        edit_distances = []
        
        # Training loop
        for epoch in range(num_epochs):
            TRAINING_ITERATIONS.inc()
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                # print(inputs.shape)
                start_time = time.time()

                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                TRAINING_LOSS.set(loss.item())
                running_loss += loss.item()
                BATCH_DURATION.observe(time.time() - start_time)
            
            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            edit_distance_sum = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Calculate edit distance (in this simple case, it's just whether prediction is correct)
                    # In real handwriting recognition with text, you'd use Levenshtein distance
                    edit_distance = (predicted != labels).float().sum().item()
                    edit_distance_sum += edit_distance
            
            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            
            accuracy = 100 * correct / total
            avg_edit_distance = edit_distance_sum / total
            edit_distances.append(avg_edit_distance)
            VALIDATION_ACCURACY.set(epoch_val_loss)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "accuracy": accuracy,
                "avg_edit_distance": avg_edit_distance
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%, "
                  f"Avg Edit Distance: {avg_edit_distance:.4f}")
        
        # Generate and log plots
        # 1. Epochs vs Training & Validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_plot.png')
        mlflow.log_artifact('loss_plot.png')
        
        # 2. Epochs vs Average Edit Distance
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), edit_distances)
        plt.xlabel('Epochs')
        plt.ylabel('Average Edit Distance')
        plt.title('Average Edit Distance per Epoch')
        plt.grid(True)
        plt.savefig('edit_distance_plot.png')
        mlflow.log_artifact('edit_distance_plot.png')
        
        # Test the model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        print(f"Test accuracy: {test_accuracy:.2f}%")
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Create a sample input for model signature
        sample_input = next(iter(test_loader))[0][:1].numpy()
        sample_output = model(torch.tensor(sample_input).to(device)).detach().cpu().numpy()
        signature = infer_signature(sample_input, sample_output)
        
        # Log the model with signature
        mlflow.pytorch.log_model(
            model, 
            "model",
            signature=signature
        )
        
        print(f"Model saved with run ID: {run.info.run_id}")
        return run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train handwriting recognition model")
    parser.add_argument("--data_dir",     type=str, default="./data", help="Directory containing the dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of test data")
    args = parser.parse_args()
    
    train_model(args.data_dir, args.train_ratio, args.val_ratio, args.test_ratio)
