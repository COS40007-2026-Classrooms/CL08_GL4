#Imports
# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Data splitting
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Utilities
import numpy as np
import pickle
import os
import pandas as pd

print("="*70)
print("PREPROCESSING NEW DATA")
print("="*70)


def pre_processing():

    #Check if data exist

    data_path = 'data/Obesity.csv'

    if not os.makedirs(data_path):
        print("The dataset path does not exist")
        return False
    
    print("="*70)
    print("Loading Data")
    print("="*70)   

    df = pd.read_csv(data_path)

    #Handle Missing Values
    print('Handling missing Values')
    df = df.dropna()

    #Remove duplicates
    print('Removing duplicate Values')
    df = df.drop_duplicates()

    #Removing Outliers

    # List of numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Detect outliers using IQR, IQR is a method that identifies outliers by calculating the interquartile range this range specifies the common values in the dataset, and any value that falls outside of this range is considered an outlier. 
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        #Selects all rows where the value in the column is an outlier, sotes col as key and the outlier rows as value in outleirs dictionary 
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        print(f"{col} outliers: {len(outliers[col])}")

        #For each numerical column, calculate IQR then remove values in df where it is outside Q1 and q3
        for col in numerical_cols:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Detect outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            # Remove outliers
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            df_cleaned = df


    #Data normalisation and scaling


    #2.1 Apply appropriate scaling to all numerical features based on their distribution
    df_cleaned.columns = df_cleaned.columns.str.strip()
    print('MTRANS' in df_cleaned.columns)  
    numerical_cols = df_cleaned.select_dtypes(include=['number']).columns
    print(numerical_cols)

    #Loops through all numerical column check in range 0 = normal, > 0.5 = Right skewed, < -0.5 = left skewed and > 1 has outleirs
    for col in df_cleaned.select_dtypes(include=['number']).columns:
        print(col, "skew:", df[col].skew())

    # Columns grouped
    standard_cols = ['Height', 'Weight', 'CH2O', 'FAF']
    minmax_cols = ['Age', 'FCVC', 'TUE']
    robust_cols = ['NCP']

    # Apply scaling, Standard scaler Makes data centred, Minmaxscaler shrinks data to 0-1 when there is skewedness and Robustscaler Scales data based on
    #Middle values and ignore outliers
    df_cleaned[standard_cols] = StandardScaler().fit_transform(df_cleaned[standard_cols])
    df_cleaned[minmax_cols] = MinMaxScaler().fit_transform(df_cleaned[minmax_cols])
    df_cleaned[robust_cols] = RobustScaler().fit_transform(df_cleaned[robust_cols])


    #2.2 Convert all non-numerical features to appropriate numerical representations.

    #Find categorical columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    print(categorical_cols)

    #Check all categories to apply encoding
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        print(f"\nColumn: {col}")
        print(df_cleaned[col].value_counts())



    #Binary encoding for only 2 values

    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

    for col in binary_cols:
        df_cleaned[col] = df_cleaned[col].map({
            'yes': 1, 'no': 0,
            'Male': 1, 'Female': 0
        })


    #Ordinal encoding For hericachical categories
    df_cleaned['CAEC'] = df['CAEC'].map({
        'no': 0,
        'Sometimes': 1,
        'Frequently': 2,
        'Always': 3
    })

    df_cleaned['CALC'] = df['CALC'].map({
        'no': 0,
        'Sometimes': 1,
        'Frequently': 2
    })


    #Nominal Encoding For no order
    df_cleaned = pd.get_dummies(df_cleaned, columns=['MTRANS'])

