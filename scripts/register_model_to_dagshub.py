import dagshub
import mlflow
import mlflow.sklearn
import joblib
import os

# -----------------------------
# 0. Hardcode Token (Bypasses Browser Login)
# -----------------------------
# PASTE YOUR PERSONAL ACCESS TOKEN HERE
dagshub.auth.add_app_token(token="731b2ee456bc3a3438b7a8353dd330618a3c624b") 

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
model_path = "Random_Forest_model.pkl"
scaler_path = "scaler.pkl"

model = joblib.load(model_path)

# -----------------------------
# 3. Log & register model + scaler
# -----------------------------
with mlflow.start_run(run_name="Random_Forest_with_Scaler"):
    
    # Log the Model into a folder named 'model'
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model", # This creates the 'model' directory
        registered_model_name="Karachi_AQI_Model"
    )
    
    # Log the Scaler into the SAME 'model' folder
    if os.path.exists(scaler_path):
        # We use artifact_path="model" to put it in the same place
        mlflow.log_artifact(scaler_path, artifact_path="model")
        print(f"✅ Scaler '{scaler_path}' logged inside the 'model' folder with the model.")
    else:
        print(f"⚠️ Warning: '{scaler_path}' not found.")

    # Optional: Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("scaler_type", "StandardScaler")

print("🚀 Model and Scaler successfully uploaded together in the 'model' directory.")