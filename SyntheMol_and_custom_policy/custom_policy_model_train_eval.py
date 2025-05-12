import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load the data
data = pd.read_csv("SyntheMol_and_custom_policy/policy_model_data.csv")

# Split into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train/validation/test split: 80/10/10
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1111, random_state=42)  # ~10% of full

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, r2))

# Display results
print(f"{'Model':<20}{'RMSE':<10}{'R2 Score':<10}")
for name, rmse, r2 in results:
    print(f"{name:<20}{rmse:<10.4f}{r2:<10.4f}")

# Find the best model by RMSE
best_model_name, best_rmse, _ = min(results, key=lambda x: x[1])
best_model = models[best_model_name]

# Save the best model to a file
joblib.dump(best_model, f"SyntheMol_and_custom_policy/{best_model_name}_policy_model.pkl")
print(f"\nSaved best model: {best_model_name} to SyntheMol_and_custom_policy/{best_model_name}_policy_model.pkl")
