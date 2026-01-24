import dagshub
import mlflow
import mlflow.sklearn
import joblib
import os

# -----------------------------
# 1. Connect to DagsHub MLflow
# -----------------------------
dagshub.init(
    repo_owner="Alihasnain388",
    repo_name="AQI_Model",
    mlflow=True
)

# -----------------------------
# 2. Load local files
# -----------------------------
# Ensure both files exist in your local directory before running
model_path = "Gradient_Boosting_model.pkl"
scaler_path = "scaler.pkl"

model = joblib.load(model_path)
# We don't need to 'load' the scaler into memory for MLflow, 
# we just need the path to the file to upload it.

# -----------------------------
# 3. Log & register model + scaler
# -----------------------------
with mlflow.start_run(run_name="Gradient_Boosting_with_Scaler"):
    
    # Log the Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )
    
    # Log the Scaler as an Artifact
    if os.path.exists(scaler_path):
        mlflow.log_artifact(scaler_path, artifact_path="model_preprocessing")
        print(f"✅ Scaler '{scaler_path}' logged as artifact.")
    else:
        print(f"⚠️ Warning: '{scaler_path}' not found. Scaler was not uploaded.")

    # Optional: Log parameters (helps for your report)
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("scaler_type", "StandardScaler")

print("🚀 Model and Scaler successfully uploaded to DagsHub!")
