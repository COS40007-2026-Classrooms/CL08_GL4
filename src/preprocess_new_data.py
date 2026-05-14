import os
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
    # 5. Encoding categorical data
    # -----------------------------
    print("Encoding categorical data...")

    # Binary mapping
    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

    binary_map = {
        'yes': 1, 'no': 0,
        'Male': 1, 'Female': 0
    }

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # Ordinal encoding
    if 'CAEC' in df.columns:
        df['CAEC'] = df['CAEC'].map({
            'no': 0,
            'Sometimes': 1,
            'Frequently': 2,
            'Always': 3
        })

    if 'CALC' in df.columns:
        df['CALC'] = df['CALC'].map({
            'no': 0,
            'Sometimes': 1,
            'Frequently': 2
        })



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