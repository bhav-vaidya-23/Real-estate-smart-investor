# 🏠 Smart Investor — Real Estate Price Predictor

A production-grade ML tool that compares **Linear Regression, Ridge, Lasso, and Random Forest** 
on the Bangalore Housing dataset, with an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
smart_investor/
├── train_models.py      ← Data cleaning + model training pipeline
├── app.py               ← Streamlit dashboard
├── requirements.txt     ← Dependencies
├── data/
│   └── Bengaluru_House_Data.csv   ← Download from Kaggle (link below)
└── models/              ← Auto-generated after training
    ├── linear_regression.pkl
    ├── ridge_regression.pkl
    ├── lasso_regression.pkl
    ├── random_forest.pkl
    ├── metrics.json
    ├── feature_columns.pkl
    └── top_locations.pkl
```

---

## 🚀 Quick Start

### 1. Download the Dataset
Get `Bengaluru_House_Data.csv` from Kaggle:
> https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

Place it in the `data/` folder.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Models
```bash
python train_models.py
```
This will:
- Load and clean the raw CSV
- Engineer features (BHK parsing, sqft normalization, outlier removal)
- Train all 4 models
- Print a comparison table of R², RMSE, MAE, and CV R²
- Save all models and metrics to `models/`

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`

---

## 🧠 ML Pipeline

### Data Cleaning
| Step | What it does |
|------|-------------|
| Drop low-utility columns | Removes `area_type`, `availability`, `society`, `balcony` |
| Parse BHK | Extracts number from "2 BHK", "3 Bedroom" etc. |
| Parse sqft | Handles ranges like "1000-1200" → takes average |
| Outlier removal | 3-sigma rule on price/sqft; bath < bhk+2; min 300 sqft/bhk |
| Location encoding | One-hot encode top 50+ locations; rare → "Other" |

### Models & Hyperparameters
| Model | Key Parameters |
|-------|---------------|
| Linear Regression | Baseline, no regularization |
| Ridge Regression | `alpha=10` (L2 regularization) |
| Lasso Regression | `alpha=0.1`, L1 regularization + feature selection |
| Random Forest | `n_estimators=100`, `max_depth=15` |

### Evaluation Metrics
- **R²** — Variance explained (higher = better, max 1.0)
- **RMSE** — Root Mean Squared Error in Lakhs (lower = better)
- **MAE** — Mean Absolute Error in Lakhs (lower = better)
- **CV R²** — 5-fold cross-validation R² (tests generalization)

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| 🔍 Property Input | Select location, sqft, BHK, bathrooms from sidebar |
| 💰 Live Prediction | Shows price from all 4 models simultaneously |
| 📊 Bar Charts | R² and RMSE comparison across models |
| 🕸️ Radar Chart | Multi-metric model comparison at a glance |
| 📈 Metrics Table | Full table with color-coded best values |
| 📋 Property Card | Summary of inputs + price-per-sqft estimate |

---

## 🔬 Technical Notes

- **Why one-hot encoding?** The dataset has 1000+ unique locations. We keep the top ~100 
  (with >10 listings) and group the rest as "Other" to avoid a massive, sparse matrix.
- **Why Ridge over Linear?** With one-hot encoded locations, multicollinearity is high. 
  Ridge regularization stabilizes coefficients significantly.
- **Why Random Forest wins?** Housing prices are non-linear (location + size interact 
  multiplicatively), which tree-based models capture naturally.
- **Cross-validation** gives a more honest estimate of generalization than train/test split alone.

---

## 📸 Expected Output (Terminal)

```
✅ Loaded dataset: 13320 rows, 9 columns
🔧 Cleaning data...
✅ Cleaned dataset: 7251 rows remaining
✅ Feature matrix shape: (7251, 242)

📊 Training & Evaluating Models...
─────────────────────────────────────────────────────────────────
  Linear Regression      | R²=0.8421 | RMSE=18.34 | MAE=9.21
  Ridge Regression       | R²=0.8489 | RMSE=17.98 | MAE=9.05
  Lasso Regression       | R²=0.8401 | RMSE=18.45 | MAE=9.33
  Random Forest          | R²=0.8912 | RMSE=15.21 | MAE=7.44
─────────────────────────────────────────────────────────────────
✅ All models & artifacts saved to models/
🏆 Best Model: Random Forest
```

---

## 🛠️ Extending the Project

- **Add XGBoost/LightGBM** — just add them to the `models` dict in `train_models.py`
- **Feature importance plot** — `model.feature_importances_` from Random Forest
- **SHAP values** — `pip install shap` and call `shap.TreeExplainer(rf_model)`  
- **Hyperparameter tuning** — wrap models with `GridSearchCV` or `RandomizedSearchCV`
- **Deploy to cloud** — `streamlit run app.py` works directly on Streamlit Community Cloud

---

*Built with ❤️ using Scikit-learn + Streamlit + Pandas*
