import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Capture credentials from the Environment (GitHub Secrets)
repo_owner = os.getenv("DAGSHUB_USERNAME")
repo_name = "AQI_Model"
token = os.getenv("DAGSHUB_TOKEN")

# 2. MANUALLY configure MLflow Authentication
# These specific variables tell MLflow to use the token as a password
os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
os.environ['MLFLOW_TRACKING_PASSWORD'] = token

# 3. Set the Tracking URI to the DagsHub MLflow endpoint
tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

print(f"🚀 Connecting directly to MLflow at: {tracking_uri}")

# 4. Load local artifacts
model_file = "Random_Forest_model.pkl"
scaler_file = "scaler.pkl"

if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ {model_file} not found. Ensure training finished correctly.")

model = joblib.load(model_file)

# 5. Log and Register
# This uses 'Basic Auth' which is silent and perfect for CI/CD
with mlflow.start_run(run_name="Automated_RF_Training_Run"):
    # Log the model and register it in the registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )

    # Log scaler inside the same model folder
    if os.path.exists(scaler_file):
        mlflow.log_artifact(scaler_file, artifact_path="model")

    mlflow.log_param("model_type", "RandomForest")
    print("✅ SUCCESS: Registered to DagsHub Model Registry!")