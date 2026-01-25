import os

# --- HARDCODED TOKEN SETUP ---
# This MUST come before importing dagshub or mlflow
os.environ['DAGSHUB_USER_TOKEN'] = "731b2ee456bc3a3438b7a8353dd330618a3c624b"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Alihasnain388"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "731b2ee456bc3a3438b7a8353dd330618a3c624b"

import dagshub
import mlflow
import mlflow.sklearn
import joblib

# 1. Connect to DagsHub MLflow
# By setting the os.environ above, this call will find the token immediately
dagshub.init(
    repo_owner="Alihasnain388",
    repo_name="AQI_Model",
    mlflow=True
)

# 2. Load local files (Saved by your training script)
model_path = "Random_Forest_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ {model_path} not found! Check training step logs.")

model = joblib.load(model_path)

# 3. Log & register model + scaler
with mlflow.start_run(run_name="Random_Forest_with_Scaler"):
    
    # Log the Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )
    
    # Log the Scaler
    if os.path.exists(scaler_path):
        mlflow.log_artifact(scaler_path, artifact_path="model")
        print(f"✅ Scaler '{scaler_path}' logged.")

    mlflow.log_param("model_type", "Random Forest")

print("🚀 SUCCESS: Model and Scaler registered on DagsHub!")