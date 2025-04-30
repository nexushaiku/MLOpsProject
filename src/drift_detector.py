# src/drift_detector.py
import mlflow
import pandas as pd
from scipy.stats import ks_2samp
import argparse
import os

PREDICTION_LOG_FILE = "prediction_logs.csv" # Should match api.py
DRIFT_THRESHOLD_P_VALUE = 0.05 # Example threshold

def check_drift(run_id, reference_artifact_path="drift_reference/reference_data.csv"):
    print(f"Checking drift against run_id: {run_id}")

    # --- 1. Load Reference Data from MLflow ---
    try:
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, reference_artifact_path)
        df_reference = pd.read_csv(local_path)
        print(f"Loaded reference data: {df_reference.shape[0]} samples")
    except Exception as e:
        print(f"Error loading reference data from MLflow run {run_id}: {e}")
        return False # Cannot perform check

    # --- 2. Load Recent Production Logs ---
    if not os.path.exists(PREDICTION_LOG_FILE):
        print("Prediction log file not found. No production data to check.")
        # Log this as no drift, or handle as needed
        mlflow.log_metric("drift_detected", 0)
        return False
    try:
        # Load recent data (e.g., last 1000 predictions)
        df_production = pd.read_csv(PREDICTION_LOG_FILE).tail(1000)
        print(f"Loaded production data: {df_production.shape[0]} samples")
    except Exception as e:
        print(f"Error loading production logs: {e}")
        return False # Cannot perform check

    if df_production.empty:
        print("No production data logged yet.")
        mlflow.log_metric("drift_detected", 0)
        return False

    # --- 3. Perform Drift Calculation (Example: KS test on features) ---
    drift_detected = False
    for col in df_reference.columns:
        if col in df_production.columns:
            ks_statistic, p_value = ks_2samp(df_reference[col], df_production[col])
            print(f"Feature '{col}': KS Stat={ks_statistic:.4f}, p-value={p_value:.4f}")
            mlflow.log_metric(f"drift_{col}_ks_stat", ks_statistic)
            mlflow.log_metric(f"drift_{col}_p_value", p_value)
            if p_value < DRIFT_THRESHOLD_P_VALUE:
                print(f" ---> Drift detected in feature: {col} (p-value < {DRIFT_THRESHOLD_P_VALUE})")
                drift_detected = True
        else:
            print(f"Warning: Feature '{col}' not found in production logs.")

    # You might also compare prediction distributions, confidence scores, etc.

    # --- 4. Log Overall Drift Status ---
    mlflow.log_metric("drift_detected", 1 if drift_detected else 0)
    print(f"Overall drift detected: {drift_detected}")
    return drift_detected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect data drift.")
    parser.add_argument("--run_id", required=True, help="MLflow run ID of the model/reference data to compare against.")
    args = parser.parse_args()

    # Optionally start a new MLflow run for the drift check itself
    with mlflow.start_run(run_name="Drift Check"):
        mlflow.log_param("checked_run_id", args.run_id)
        drift_status = check_drift(args.run_id)
        # Exit with non-zero code if drift detected (can be used by orchestrator)
        exit(1 if drift_status else 0)

