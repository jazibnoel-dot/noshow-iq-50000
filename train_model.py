from noshow_iq.preprocess import load_and_clean, get_features_and_target
from noshow_iq.model import train
from dotenv import load_dotenv
import os

load_dotenv()

df = load_and_clean("noshow_iq/data/KaggleV2-May-2016.csv")
X, y = get_features_and_target(df)
metrics = train(X, y)

print("Training complete!")
print(metrics)

# Save training run to MongoDB ONLY if URI is available
# (build environments may not have MongoDB access)
mongo_uri = os.getenv("MONGO_URI")
if mongo_uri and mongo_uri.startswith("mongodb"):
    try:
        from pymongo import MongoClient
        from datetime import datetime, timezone
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["noshowiq"]
        db["training_runs"].insert_one({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_size": metrics["training_size"],
            "precision_0": metrics["precision_0"],
            "recall_0": metrics["recall_0"],
            "f1_0": metrics["f1_0"],
            "precision_1": metrics["precision_1"],
            "recall_1": metrics["recall_1"],
            "f1_1": metrics["f1_1"],
            "imbalance_technique": metrics["imbalance_technique"],
        })
        print("Training run saved to MongoDB!")
    except Exception as e:
        print(f"MongoDB save skipped: {e}")
else:
    print("No MONGO_URI — skipping MongoDB save.")
