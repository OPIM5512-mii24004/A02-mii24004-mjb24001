# Assignment 2: Neural Network Regression Analysis

## Project Overview
This repository contains the materials for assignment **A02**, completed by **Anwar and Madi**. The project focuses on training a neural network regression model using the California Housing dataset to predict median house values.

## Model Implementation Details
To ensure high performance and prevent overfitting, the following configurations were implemented:
* **Architecture:** A Multi-Layer Perceptron (MLP) with two hidden layers (10 and 5 neurons respectively).
* **Activation Function:** `relu`.
* **Overfitting Control:** Early-stopping is enabled, using a 20% validation fraction from the training data.
* **Data Strategy:** The dataset is split into training, validation, and testing sets. Features are scaled using `StandardScaler` to ensure convergence.



## Data Used
* **Dataset:** California Housing dataset (sourced from Scikit-Learn).
* **Target Variable:** `MedHouseVal` (Median house value for California districts).

---

## How to Run the Script
Follow these steps to reproduce the results:

### 1. Install the requirements
```bash
pip install -r requirements.txt

### 2. Change your working directory
```bash
cd A02-mii24004-mjb24001

### 3. Run the code
```bash
python scripts/NN_regression.py

---

## Expected Output

### Figures
The script automatically creates a `figures/` directory and saves the following plots:
* **`train_plot.png`**: A scatter plot comparing predicted vs. actual values for the training set.
* **`test_plot.png`**: A scatter plot comparing predicted vs. actual values for the final test set.


### Metrics
The following performance metrics are printed to the console for both the training and testing phases:
* **Coefficient of Determination ($R^2$)**
* **Mean Absolute Error (MAE)**
* **Mean Absolute Percentage Error (MAPE)**

---

## Performance Summary
The model demonstrates strong generalization, with the test metrics remaining very close to the training metrics.

| Metric | Training Set | Testing Set |
| :--- | :--- | :--- |
| **$R^2$ Score** | 0.762 | 0.744 |
| **MAE** | 0.397 | 0.404 |
| **MAPE** | 0.224 | 0.232 |
