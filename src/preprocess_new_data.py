import os

# Pull dataset from DVC
os.system("dvc pull")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


print("="*70)
print("PREPROCESSING NEW DATA")
print("="*70)


def pre_processing():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    data_path = "data/Obesity.csv"

    if not os.path.exists(data_path):
        print("Dataset not found!")
        return False

    print("Loading data...")
    df = pd.read_csv(data_path)

    # -----------------------------
    # 2. Missing values
    # -----------------------------
    print("Handling missing values...")
    df = df.dropna()

    # -----------------------------
    # 3. Remove duplicates
    # -----------------------------
    print("Removing duplicates...")
    df = df.drop_duplicates()

    # -----------------------------
    # 4. Outlier removal (IQR)
    # -----------------------------
    print("Removing outliers...")

    numerical_cols = df.select_dtypes(include=["number"]).columns

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # -----------------------------
    # 5. Load feature columns
    # -----------------------------
    print("Encoding categorical data...")

    # Binary mapping
    feature_columns_path = 'artifacts/preprocessing/feature_columns.json'
    if not os.path.exists(feature_columns_path):
        feature_columns_path = 'artifacts/feature_columns.json'
        if not os.path.exists(feature_columns_path):
            print(f"Feature columns not found at {feature_columns_path}")
            return False

    with open(feature_columns_path, 'r') as f:
        feature_columns = json.load(f)
    print(f"Loaded {len(feature_columns)} feature columns")

    # 4. Extract target column
    print("\n Processing data...")
    if 'target' in df.columns:
        y_new = df['target'].values
        X_df = df.drop('target', axis=1)
    elif 'y' in df.columns:
        y_new = df['y'].values
        X_df = df.drop('y', axis=1)
    else:
        y_new = df.iloc[:, -1].values
        X_df = df.iloc[:, :-1]

    print(f"Target shape: {y_new.shape}")
 



    # One-hot encoding
    df = pd.get_dummies(df, columns=['MTRANS'], drop_first=True)


    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    print("Creating new features...")

    # 1. BMI (Body Mass Index)
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # 2. Activity score (physical activity - screen time)
    if 'FAF' in df.columns and 'TUE' in df.columns:
        df['Activity_Score'] = df['FAF'] - df['TUE']

    # 3. Eating behavior score
    if 'FCVC' in df.columns and 'NCP' in df.columns:
        df['Eating_Behavior'] = df['FCVC'] + df['NCP']

    print("Feature engineering complete!")

    # -----------------------------
    # 6. Split features/target
    # -----------------------------
    target = "NObeyesdad"

    X = df.drop(columns=[target])
    y = df[target]

    le = LabelEncoder()
    y = le.fit_transform(y)

    # -----------------------------
    # 7. Train/test split
    # -----------------------------
    print("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # 8. Scaling
    # -----------------------------
    print("Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # 9. Save artifacts
    # -----------------------------
    print("Saving processed data...")

    os.makedirs("artifacts/data", exist_ok=True)
    os.makedirs("artifacts/preprocessing", exist_ok=True)

    np.save("artifacts/data/X_train.npy", X_train_scaled)
    np.save("artifacts/data/X_test.npy", X_test_scaled)
    np.save("artifacts/data/y_train.npy", y_train)
    np.save("artifacts/data/y_test.npy", y_test)

    joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")

    print("Preprocessing complete!")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    pre_processing()
