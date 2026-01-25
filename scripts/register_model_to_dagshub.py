import dagshub
import mlflow
import mlflow.sklearn
import joblib
import os

# 1. FORCE non-interactive mode
# This tells DagsHub to use these tokens instead of asking for a browser
token = os.getenv("DAGSHUB_TOKEN")
username = os.getenv("DAGSHUB_USERNAME")

os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = token

# 2. Set the tracking URI MANUALLY (This is the "Silent" way)
mlflow.set_tracking_uri(f"https://dagshub.com/{username}/AQI_Model.mlflow")

# 3. Init DagsHub - Use 'force=True' to override any cached logins
dagshub.init(
    repo_owner=username,
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