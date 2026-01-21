import pandas as pd
from pymongo import MongoClient

# ---------------------------------------------------------
# 1. SETTINGS: Replace <password> and <cluster-url> 
# with your actual string from MongoDB Atlas
# ---------------------------------------------------------
MONGO_URI = "mongodb+srv://ali321hasnain_db_user:etRWe1e6ASFlpwEO@cluster0.1eklm6h.mongodb.net/?appName=Cluster0"
DB_NAME = "AQIPredictionSystem"
COLLECTION_NAME = "karachi_features"

def upload_to_mongodb():
    file_name = 'karachi_final_features.csv'
    
    print(f"📖 Reading {file_name}...")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"❌ Error: {file_name} not found in this folder!")
        return

    # Convert the CSV rows into a format MongoDB understands (JSON/Dict)
    records = df.to_dict('records')

    print("🔌 Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Clean old data to avoid duplicates
        print("🧹 Clearing previous records...")
        collection.delete_many({})

        # Upload everything
        print(f"⬆️ Uploading {len(records)} records...")
        result = collection.insert_many(records)
        
        print(f"✅ SUCCESS! {len(result.inserted_ids)} features are now in your cloud Feature Store.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    upload_to_mongodb()