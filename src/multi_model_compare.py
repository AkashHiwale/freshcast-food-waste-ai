import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: XGBoost if installed
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False

data = pd.read_csv("../data/food_waste_dataset.csv")
data = pd.get_dummies(data)            # convert categories
X = data.drop("wasted_food", axis=1)
y = data["wasted_food"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge()),
    "Lasso": make_pipeline(StandardScaler(), Lasso()),
    "ElasticNet": make_pipeline(StandardScaler(), ElasticNet()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "SVR": make_pipeline(StandardScaler(), SVR()),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
}
if has_xgb:
    models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

def eval_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:12s} | RMSE: {rmse:8.4f} | MAE: {mae:8.4f} | R2: {r2:6.4f}")

print("Training and evaluating models...\n")
for name, model in models.items():
    eval_model(name, model, X_train, X_test, y_train, y_test)

# Optional: show top feature importances from tree model (RandomForest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
print("\nTop feature importances (RandomForest):")
print(importances)
