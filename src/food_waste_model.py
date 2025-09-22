# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Step 2: Load dataset
data = pd.read_csv("../data/food_waste_dataset.csv")

# Convert categorical columns (e.g., 'Cake', 'Bread', etc.) to numbers
data = pd.get_dummies(data)

# Step 3: Split features (X) and target (y)
X = data.drop("wasted_food", axis=1)   # all columns except target
y = data["wasted_food"]                # target column

# Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (important for MLP and Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)
}

# Train models and evaluate
results = {}
for name, model in models.items():
    # Choose scaled or unscaled data
    X_tr, X_te = (X_train_scaled, X_test_scaled) if name in ["Ridge Regression", "MLP Regressor"] else (X_train, X_test)
    
    # Train and predict
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

#Save the trained model to a .pkl file
with open("linear_model.pkl", "wb") as f:
    pickle.dump(models["Linear Regression"], f)

print("Linear Regression model saved as linear_model.pkl")

# Print results
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.2f}")

# Plot actual vs predicted for each model
plt.figure(figsize=(12, 8))
for i, (name, model) in enumerate(models.items(), 1):
    plt.subplot(2, 2, i)
    X_te = X_test_scaled if name in ["Ridge Regression", "MLP Regressor"] else X_test
    y_pred = model.predict(X_te)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(name)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
plt.tight_layout()
plt.show()