import os
import dagshub
import mlflow
import mlflow.sklearn
import joblib

# 1. Provide credentials to environment so it doesn't ask for a browser login
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")

# 2. Init DagsHub (it will now see the environment variables and skip the prompt)
dagshub.init(
    repo_owner="Alihasnain388",
    repo_name="AQI_Model",
    mlflow=True
)

# -----------------------------
# 3. Load local model + scaler
# -----------------------------
# Verify files exist before loading
if not os.path.exists("Random_Forest_model.pkl") or not os.path.exists("scaler.pkl"):
    raise FileNotFoundError("Critical model files (.pkl) are missing from the runner!")

model = joblib.load("Random_Forest_model.pkl")
scaler_path = "scaler.pkl"

# -----------------------------
# 4. Log EVERYTHING properly
# -----------------------------
with mlflow.start_run(run_name="AQI_Model_FINAL_UPLOAD"):

    # Log the model and register it in the DagsHub Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )

    # Log scaler INSIDE model folder so they stay together
    mlflow.log_artifact(
        scaler_path,
        artifact_path="model"
    )

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("scaler", "StandardScaler")

print("✅ Model + scaler uploaded and registered to DagsHub successfully!")