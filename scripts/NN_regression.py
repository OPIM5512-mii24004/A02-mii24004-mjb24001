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

# Split train and test

# train
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# test and val
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled


#PR #2: Add MLPRegressor with early stopping

mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(10,5),
                   max_iter=200,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) # important!
mlp.fit(X_train_scaled, y_train)


# Predict on train, validation, and test

y_pred_train = mlp.predict(X_train_scaled)
y_pred_val   = mlp.predict(X_val_scaled)
y_pred_test  = mlp.predict(X_test_scaled)

# Define plot function 
def scatter_with_reference(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1, color='red')  # reference line
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot training predictions vs actual
scatter_with_reference(y_train, y_pred_train, "Predicted vs Actual — Train")
