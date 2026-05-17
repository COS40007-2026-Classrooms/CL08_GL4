import numpy as np
import joblib
import os
import json
from datetime import datetime

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


print("="*60)
print("MODEL TRAINING - SVM vs Decision Tree")
print("="*60)

#Directories
os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/preprocessing", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("artifacts/data", exist_ok=True)


#Data Loading
X_train = np.load("artifacts/data/X_train.npy")
X_test = np.load("artifacts/data/X_test.npy")
y_train = np.load("artifacts/data/y_train.npy")
y_test = np.load("artifacts/data/y_test.npy")

print('Data Loaded Succesfully')


# ────────────────────────────────────
# Feature Columns Metadata
# ────────────────────────────────────
#Saving feature columns
num_features = X_train.shape[1]
feature_columns = [f"feature_{i}" for i in range(num_features)]

with open("artifacts/preprocessing/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=4)
print(f"\nSaved feature_columns.json ({num_features} features)")

# ────────────────────────────────────
# Target Column Info
# ────────────────────────────────────
target_info = {
    "target_column": "obesity_class",
    "num_classes": len(np.unique(y_train)),
    "unique_classes": sorted([int(c) for c in np.unique(y_train)])
}
with open("artifacts/preprocessing/target_column.json", "w") as f:
    json.dump(target_info, f, indent=4)
print(f"Saved target_column.json ({target_info['num_classes']} classes)")

# ────────────────────────────────────
# Create and Save Scaler
# ────────────────────────────────────
print("\nCreating StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")
print("Saved scaler.pkl")

# Save scaled data (backup/reference for downstream use)
np.save("artifacts/data/X_train_scaled.npy", X_train_scaled)
np.save("artifacts/data/X_test_scaled.npy", X_test_scaled)
print("Saved X_train_scaled.npy and X_test_scaled.npy")

# ────────────────────────────────────
#SVM Model
# ────────────────────────────────────
print('\n Training SVM Model')

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
svm_precision = precision_score(y_test, svm_pred, average='weighted')
svm_recall = recall_score(y_test, svm_pred, average='weighted')


print(f"SVM Accuracy: {svm_acc:.2f}")
print(f"SVM F1-score: {svm_f1:.2f}")

# ────────────────────────────────────
#Decision Tree
# ────────────────────────────────────
print('\n Training Decision Tree Model')

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average="weighted")
dt_precision = precision_score(y_test, dt_pred, average='weighted')
dt_recall = recall_score(y_test, dt_pred, average='weighted')

print(f"Decision Tree Accuracy: {dt_acc:.2f}")
print(f"Decision Tree F1-score: {dt_f1:.2f}")

#Selecting model
print('\n Model comparison Process')

if svm_acc >= dt_acc:
    best_model = svm_model
    best_name = "SVM"
    best_acc = svm_acc
    best_f1 = svm_f1
    best_precision = svm_precision
    best_recall = svm_recall
else:
    best_model = dt_model
    best_name = "Decision Tree"
    best_acc = dt_acc
    best_f1 = dt_f1
    best_precision = dt_precision
    best_recall = dt_recall

print(f"Best Model: {best_name}")

# -------------------------
# 5. Save best model
# -------------------------
os.makedirs("artifacts/models", exist_ok=True)

joblib.dump(best_model, "artifacts/models/best_model.pkl")

print("\nModel saved successfully!")
print("="*60)

# -------------------------
# 6. Saving test metrics
# -------------------------
test_metrics = {
    "model_name": best_name,
    "accuracy": float(best_acc),
    "f1_score": float(best_f1),
    "precision": float(best_precision),
    "recall": float(best_recall),
    "test_samples": int(len(y_test)),
    "timestamp": datetime.now().isoformat()
}

with open("artifacts/metrics/test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=4)
print("Saved test_metrics.json")

# ────────────────────────────────────
# Save Model Metadata
# ────────────────────────────────────
model_metadata = {
    "model_type": best_name,
    "model_name": best_name.lower().replace(" ", "_"),
    "num_features": num_features,
    "num_classes": target_info["num_classes"],
    "feature_columns": feature_columns,
    "target_column": target_info["target_column"],
    "training_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "training_completed": datetime.now().isoformat(),
    "hyperparameters": {
        "model": best_name,
        "svm": {"kernel": "rbf", "C": 1, "gamma": "scale"} if best_name == "SVM" else None,
        "decision_tree": {"max_depth": 10, "random_state": 42} if best_name == "Decision Tree" else None
    },
    "test_performance": {
        "accuracy": float(best_acc),
        "f1_score": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall)
    },
    "data_info": {
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "features_count": num_features,
        "classes_count": target_info["num_classes"],
        "target_min": int(np.min(y_test)),
        "target_max": int(np.max(y_test)),
        "target_mean": float(np.mean(y_test)),
        "target_std": float(np.std(y_test))
    }
}

with open("artifacts/metadata/model_info.json", "w") as f:
    json.dump(model_metadata, f, indent=4)
print("Saved model_info.json")