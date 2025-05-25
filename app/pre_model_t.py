import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import joblib

def clean_sizes(s):
    try:
        sizes = eval(s) if isinstance(s, str) else []
        return [x.strip().upper().replace("LARGE", "L").replace("MEDIUM", "M").replace("SMALL", "S") for x in sizes]
    except:
        return []

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["SellPrice"])
    df["Sizes_list_cleaned"] = df["Sizes"].apply(clean_sizes)
    df["Sizes_str"] = df["Sizes_list_cleaned"].apply(lambda x: " ".join(x))
    df = df[["BrandName", "Category", "MRP", "Details", "Sizes_str", "SellPrice"]]
    return df

def build_preprocessor():
    return ColumnTransformer([
        ("brand", OneHotEncoder(handle_unknown="ignore"), ["BrandName"]),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["Category"]),
        ("mrp", StandardScaler(), ["MRP"]),
        ("details", TfidfVectorizer(max_features=100), "Details"),
        ("sizes", TfidfVectorizer(max_features=50), "Sizes_str")
    ])

def train_models(X_train, y_train, X_test, y_test, preprocessor):
    models = {
        "xgb": XGBRegressor(objective="reg:squarederror", random_state=42),
        "rf": RandomForestRegressor(random_state=42),
        "lgbm": LGBMRegressor(random_state=42)
    }

    param_grid = {
        "xgb": {"model__n_estimators": [100], "model__max_depth": [5]},
        "rf": {"model__n_estimators": [100], "model__max_depth": [20]},
        "lgbm": {"model__n_estimators": [100], "model__num_leaves": [31]}
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name.upper()} model...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=50)),
            ("model", model)
        ])
        
        grid = GridSearchCV(pipeline, param_grid[name], cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "best_params": grid.best_params_,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

        joblib.dump(best_model, f"{name}_model.pkl")
        print(f"Saved {name} model to {name}_model.pkl")
        print(f"{name.upper()} MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    return results

def main():
    data_path = os.path.join("data", "EDA_Cleaned_FashionDataset.csv")
    df = load_and_prepare_data(data_path)

    X = df.drop(columns=["SellPrice"])
    y = df["SellPrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    results = train_models(X_train, y_train, X_test, y_test, preprocessor)

    print("\nFinal Results:")
    for model_name, res in results.items():
        print(f"{model_name.upper()} -> MSE: {res['mse']:.2f}, MAE: {res['mae']:.2f}, RMSE: {res['rmse']:.2f}, R2: {res['r2']:.2f}, Best Params: {res['best_params']}")
    
if __name__ == "__main__":
    main()
    
    

# Final Results:
# XGB -> MSE: 48220.20, MAE: 149.55, RMSE: 219.59, R2: 0.90, Best Params: {'model__max_depth': 5, 'model__n_estimators': 100}
# RF -> MSE: 45949.27, MAE: 138.53, RMSE: 214.36, R2: 0.90, Best Params: {'model__max_depth': 20, 'model__n_estimators': 100}
# LGBM -> MSE: 48198.31, MAE: 153.30, RMSE: 219.54, R2: 0.90, Best Params: {'model__n_estimators': 100, 'model__num_leaves': 31}