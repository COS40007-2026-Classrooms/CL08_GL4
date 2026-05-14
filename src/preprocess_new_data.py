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

    #Encode Categorical Features