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

#SVM Model
print('\n Training SVM Model')

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')

print(f"SVM Accuracy: {svm_acc:.2f}")
print(f"SVM F1-score: {svm_f1:.2f}")

#Decision Tree
print('\n Training Decision Tree Model')

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average="weighted")

print(f"Decision Tree Accuracy: {dt_acc:.2f}")
print(f"Decision Tree F1-score: {dt_f1:.2f}")

#Selecting model
print('\n Model comparison Process')

if svm_acc >= dt_acc:
    best_model = svm_model
    best_name = "SVM"
else:
    best_model = dt_model
    best_name = "Decision Tree"

print(f"Best Model: {best_name}")

# -------------------------
# 5. Save best model
# -------------------------
os.makedirs("artifacts/models", exist_ok=True)

joblib.dump(best_model, "artifacts/models/best_model.pkl")

print("\nModel saved successfully!")
print("="*60)