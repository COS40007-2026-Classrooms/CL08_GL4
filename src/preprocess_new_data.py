import os
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


print("=" * 70)
print("PREPROCESSING (NEW DATA)")
print("=" * 70)


def load_feature_columns():
    paths = [
        "artifacts/preprocessing/feature_columns.json",
        "artifacts/feature_columns.json"
    ]

    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)

    return None


def pre_processing():


    path = "data/new_data.csv"

    if not os.path.exists(path):
        print("Dataset not found!")
        return False

    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")

    # -----------------------------
    # 2. Clean
    # -----------------------------
    df = df.dropna().drop_duplicates()

    # outlier removal (optional safe)
    num_cols = df.select_dtypes(include=["number"]).columns

    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    # -----------------------------
    # 3. Load feature schema (IMPORTANT)
    # -----------------------------
    feature_columns = load_feature_columns()

    if not feature_columns:
        print("Feature schema missing!")
        return False

    # -----------------------------
    # 4. Detect target column flexibly
    # -----------------------------
    target_candidates = ["NObeyesdad", "target", "y"]

    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break

    if not target_col:
        target_col = df.columns[-1]  # fallback

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # -----------------------------
    # 5. ALIGN FEATURES (KEY PART)
    # -----------------------------
    print("Aligning features to training schema...")

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0  # missing column
        else:
            X[col] = X[col]

    # drop extra columns
    X = X[feature_columns]

    # handle non-numeric safely
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -----------------------------
    # 6. Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=0.2,
        random_state=42,
    )

    # -----------------------------
    # 7. Scale
    # -----------------------------
    scaler = StandardScaler()
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    # Columns grouped
    standard_cols = ['Height', 'Weight', 'CH2O', 'FAF']
    minmax_cols = ['Age', 'FCVC', 'TUE']
    robust_cols = ['NCP']

    # Apply scaling, Standard scaler Makes data centred, Minmaxscaler shrinks data to 0-1 when there is skewedness and Robustscaler Scales data based on
    #Middle values and ignore outliers
    df[standard_cols] = StandardScaler().fit_transform(df[standard_cols])
    df[minmax_cols] = MinMaxScaler().fit_transform(df[minmax_cols])
    df[robust_cols] = RobustScaler().fit_transform(df[robust_cols])

    #Ordinal encoding For hericachical categories
    df['CAEC'] = df['CAEC'].map({
        'no': 0,
        'Sometimes': 1,
        'Frequently': 2,
        'Always': 3
    })

    df['CALC'] = df['CALC'].map({
        'no': 0,
        'Sometimes': 1,
        'Frequently': 2
    })

    #Binary encoding for only 2 values

    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

    for col in binary_cols:
        df[col] = df[col].map({
            'yes': 1, 'no': 0,
            'Male': 1, 'Female': 0
        })

    #Nominal Encoding For no order
    df = pd.get_dummies(df, columns=['MTRANS'])

    # -----------------------------
    # 8. Save
    # -----------------------------
    os.makedirs("artifacts/data", exist_ok=True)
    os.makedirs("artifacts/preprocessing", exist_ok=True)

    np.save("artifacts/data/X_train.npy", X_train)
    np.save("artifacts/data/X_test.npy", X_test)
    np.save("artifacts/data/y_train.npy", y_train)
    np.save("artifacts/data/y_test.npy", y_test)

    joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")

    print("Done preprocessing!")
    return True


if __name__ == "__main__":
    pre_processing()