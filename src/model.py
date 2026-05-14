import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


print("="*60)
print("MODEL TRAINING - SVM vs Decision Tree")
print("="*60)

#Data Loading
X_train = np.load("artifacts/data/X_train.npy")
X_test = np.load("artifacts/data/X_test.npy")
y_train = np.load("artifacts/data/y_train.npy")
y_test = np.load("artifacts/data/y_test.npy")

print('Data Loaded Succesfully')

#SVM Model
print('\n Training SVM Model')

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')

print(f"SVM Accuracy: {svm_acc:.2f}")
print(f"SVM F1-score: {svm_f1:.2f}")

