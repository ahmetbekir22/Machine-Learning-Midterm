import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

np.random.seed(42)

# Task 1 
# Generating synthetic data with 5 features
n_samples = 5000
n_features = 5
data = np.random.uniform(-2, 2, size=(n_samples, n_features))

# Creating dependencies between features to simulate relationships
data[:, 2] = 0.5 * data[:, 0] + 0.3 * data[:, 1] + np.random.normal(0, 0.2, n_samples)
data[:, 3] = 0.7 * data[:, 2] - 0.2 * data[:, 1] + np.random.normal(0, 0.3, n_samples)
data[:, 4] = -0.4 * data[:, 2] + 0.6 * data[:, 1] + np.random.normal(0, 0.15, n_samples)

# Introduce missing values into 2% of the data for feature 2
missing_percentage = 0.02
n_missing = int(n_samples * missing_percentage)
missing_indices = np.random.choice(n_samples, n_missing, replace=False)

# Create a copy of the data with missing values in Feature 2
data_with_missing = data.copy()
data_with_missing[missing_indices, 2] = np.nan

# Random Imputation: Replace missing values in Feature 2 with random values
data_random_imputed = data_with_missing.copy()
data_random_imputed[np.isnan(data_random_imputed[:, 2]), 2] = np.random.uniform(
    np.nanmin(data[:, 2]), np.nanmax(data[:, 2]), size=np.sum(np.isnan(data_with_missing[:, 2]))
)

# Split data into missing and non-missing for regression imputation
not_missing = ~np.isnan(data_with_missing[:, 2])  # Indices where data is not missing
missing = np.isnan(data_with_missing[:, 2])  # Indices where data is missing

X_train = data_with_missing[not_missing][:, [0, 1, 3, 4]]  # Training features
y_train = data_with_missing[not_missing, 2]  # Training target
X_test = data_with_missing[missing][:, [0, 1, 3, 4]]  # Features for missing values

# Regression Imputation: Predict missing values using Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

data_regression_imputed = data_with_missing.copy()
data_regression_imputed[missing, 2] = regressor.predict(X_test)

# Save datasets for later comparison
datasets = {
    "original_data": data,
    "data_with_missing": data_with_missing,
    "data_random_imputed": data_random_imputed,
    "data_regression_imputed": data_regression_imputed,
}

for name, dataset in datasets.items():
    pd.DataFrame(dataset, columns=["Feature1", "Feature2", "Target1", "Target2", "Target3"]).to_csv(f"{name}.csv", index=False)

# Calculate Mean Squared Error (MSE) for Task 1
mse_target1 = mean_squared_error(data[missing_indices, 2], data_regression_imputed[missing_indices, 2])
mse_target2 = mean_squared_error(
    data[missing_indices, 3], 0.7 * data_regression_imputed[missing_indices, 2] - 0.2 * data[missing_indices, 1]
)
mse_target3 = mean_squared_error(
    data[missing_indices, 4], -0.4 * data_regression_imputed[missing_indices, 2] + 0.6 * data[missing_indices, 1]
)

print("Task 1 - Mean Squared Error (MSE):")
print(f"Target 1: {mse_target1:.3f}")
print(f"Target 2: {mse_target2:.3f}")
print(f"Target 3: {mse_target3:.3f}")

# Visualization of actual vs predicted values
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(data[missing_indices, 0], data[missing_indices, 2], color='blue', marker='*', label='Actual Value')
plt.scatter(data[missing_indices, 0], data_regression_imputed[missing_indices, 2], color='red', marker='*', label='Predicted Value')
plt.title("Target 1")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(data[missing_indices, 2], data[missing_indices, 3], color='blue', marker='*', label='Actual Value')
plt.scatter(data_regression_imputed[missing_indices, 2], data[missing_indices, 3], color='red', marker='*', label='Predicted Value')
plt.title("Target 2")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(data[missing_indices, 2], data[missing_indices, 4], color='blue', marker='*', label='Actual Value')
plt.scatter(data_regression_imputed[missing_indices, 2], data[missing_indices, 4], color='red', marker='*', label='Predicted Value')
plt.title("Target 3")
plt.legend()

plt.tight_layout()
plt.show()

# Task 2 
datasets = {
    "Original": pd.read_csv("original_data.csv").values,
    "Random Imputation": pd.read_csv("data_random_imputed.csv").values,
    "Regression Imputation": pd.read_csv("data_regression_imputed.csv").values,
}

results = {}
for name, dataset in datasets.items():
    X = dataset[:, [0, 1, 3, 4]]  # Features
    y = dataset[:, 2]  # Target (Feature2)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Neural Network (MLP Regressor)
    model = MLPRegressor(hidden_layer_sizes=(10,), random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    # Predict and calculate MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse

print("\nTask 2 - Mean Squared Error (MSE):")
for name, mse in results.items():
    print(f"{name}: {mse:.3f}")
