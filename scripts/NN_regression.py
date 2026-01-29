# Import necessary packages

import warnings
warnings.filterwarnings("ignore")  # keep output clean; MLP may warn about convergence

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import California housing data

data = fetch_california_housing(as_frame=True)
print(data.DESCR)  # dataset description
X = data.frame.drop(columns=["MedHouseVal"])
y = data.frame["MedHouseVal"]

#Split train and test

# train
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# test and val
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)