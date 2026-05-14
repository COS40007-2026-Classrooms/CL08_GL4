import os
import sys
import json
import numpy as np
import pandas as pd

print("="*70)
print("PREPROCESSING NEW DATA")
print("="*70)

def pre_processing():

    #Check if data exist

    data_path = 'data/Obesity.csv'