import os
import json
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score


print("=" * 60)
print("MODEL MONITORING")
print("=" * 60)

# Load test data
X_test = np.load("artifacts/data/X_test.npy")
y_test = np.load("artifacts/data/y_test.npy")

# Load trained model
model = joblib.load("artifacts/models/best_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Calculate monitoring metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Current Accuracy: {accuracy:.4f}")
print(f"Current F1-score: {f1:.4f}")

# Simple monitoring threshold
accuracy_threshold = 0.80

if accuracy < accuracy_threshold:
    status = "WARNING: Model performance dropped. Retraining may be required."
else:
    status = "Model performance is acceptable."

print(status)

# Save monitoring results
os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("logs", exist_ok=True)

monitoring_results = {
    "accuracy": accuracy,
    "f1_score": f1,
    "threshold": accuracy_threshold,
    "status": status
}

with open("artifacts/metrics/monitoring_metrics.json", "w") as f:
    json.dump(monitoring_results, f, indent=4)

with open("logs/monitoring.log", "w") as f:
    f.write("MODEL MONITORING REPORT\n")
    f.write("=======================\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Threshold: {accuracy_threshold}\n")
    f.write(f"Status: {status}\n")

print("Monitoring results saved successfully.")