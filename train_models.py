"""
Smart Investor - Model Training Pipeline
Trains Linear Regression, Ridge, Lasso, and Random Forest
on the Bangalore Housing dataset and saves models + metrics.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
def load_data(path="C:\\Users\\vaidy\\OneDrive\\Desktop\\Programming\\mlproject\\Bengaluru_House_Data.csv"):
    """Load the Bangalore housing dataset."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 Cleaning data...")

    # Drop columns with too many nulls or low utility
    df = df.drop(columns=["area_type", "availability", "society", "balcony"], errors="ignore")

    # Drop rows missing key fields
    df.dropna(subset=["location", "size", "bath", "price"], inplace=True)

    # --- Parse BHK from 'size' column (e.g. "2 BHK", "4 Bedroom") ---
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df.drop(columns=["size"], inplace=True)

    # --- Parse total_sqft (handle ranges like "1000-1200") ---
    def convert_sqft(x):
        try:
            if isinstance(x, str) and "-" in x:
                parts = x.split("-")
                return (float(parts[0]) + float(parts[1])) / 2
            return float(x)
        except:
            return np.nan

    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

    # Drop rows where sqft conversion failed
    df.dropna(subset=["total_sqft", "bhk"], inplace=True)

    # --- Feature Engineering ---
    df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

    # Remove extreme outliers (price per sqft)
    mean_pps = df["price_per_sqft"].mean()
    std_pps = df["price_per_sqft"].std()
    df = df[(df["price_per_sqft"] > mean_pps - 3 * std_pps) &
            (df["price_per_sqft"] < mean_pps + 3 * std_pps)]

    # Remove bhk outliers
    df = df[df["bhk"] <= 10]
    df = df[df["bath"] < df["bhk"] + 2]

    # Remove sqft outliers (min 300 sqft per BHK)
    df = df[df["total_sqft"] / df["bhk"] >= 300]

    # --- Location encoding ---
    # Keep top 50 locations; rest → "Other"
    location_counts = df["location"].value_counts()
    top_locations = location_counts[location_counts > 10].index.tolist()
    df["location"] = df["location"].apply(
        lambda x: x.strip() if x.strip() in top_locations else "Other"
    )

    df.drop(columns=["price_per_sqft"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ Cleaned dataset: {df.shape[0]} rows remaining")
    return df, top_locations


# ─────────────────────────────────────────────
# 3. FEATURE PREPARATION
# ─────────────────────────────────────────────
def prepare_features(df: pd.DataFrame):
    """One-hot encode location and return X, y."""
    X = pd.get_dummies(df.drop(columns=["price"]), columns=["location"])
    y = df["price"]  # price in Lakhs
    print(f"✅ Feature matrix shape: {X.shape}")
    return X, y


# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
def train_and_evaluate(X, y):
    """Train all models and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    trained_models = {}

    print("\n📊 Training & Evaluating Models...")
    print("─" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse  = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        # Cross-val R² (5-fold)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

        results[name] = {
            "MSE":  round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE":  round(mae, 4),
            "R2":   round(r2, 4),
            "CV_R2": round(cv_r2, 4),
        }
        trained_models[name] = model

        print(f"  {name:<22} | R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

    print("─" * 65)
    return trained_models, results, list(X.columns)


# ─────────────────────────────────────────────
# 5. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(trained_models, results, feature_cols, top_locations):
    os.makedirs("models", exist_ok=True)

    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        with open(f"models/{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("models/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    with open("models/top_locations.pkl", "wb") as f:
        pickle.dump(top_locations, f)

    print("\n✅ All models & artifacts saved to models/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df, top_locations = clean_data(df)
    X, y = prepare_features(df)
    trained_models, results, feature_cols = train_and_evaluate(X, y)
    save_artifacts(trained_models, results, feature_cols, top_locations)

    print("\n🏆 Best Model:", max(results, key=lambda k: results[k]["R2"]))
    print("   Run `streamlit run app.py` to launch the dashboard!")
