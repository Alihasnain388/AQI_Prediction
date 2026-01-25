import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Capture credentials from the Environment (GitHub Secrets)
repo_owner = os.getenv("DAGSHUB_USERNAME")
repo_name = "AQI_Model"
token = os.getenv("DAGSHUB_TOKEN")

# 2. MANUALLY configure MLflow (The "Silent" Way)
# This forces MLflow to use your token and skip the DagsHub login prompt
os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
os.environ['MLFLOW_TRACKING_PASSWORD'] = token
tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

print(f"🚀 Connecting to DagsHub MLflow at: {tracking_uri}")

# 3. Load local artifacts saved by the training script
model_path = "Random_Forest_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ {model_path} not found! Check training logs.")

model = joblib.load(model_path)

# 4. Log and Register
with mlflow.start_run(run_name="Automated_RF_Training"):
    # Log the model and register it in the DagsHub Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )

    # Log scaler inside the same folder
    if os.path.exists(scaler_path):
        mlflow.log_artifact(scaler_path, artifact_path="model")

    mlflow.log_param("model_type", "RandomForest")
    print("✅ Model + Scaler registered to DagsHub successfully!")