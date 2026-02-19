"""
Smart Investor — Real Estate Price Predictor Dashboard
Deployable on Streamlit Community Cloud (no CSV needed — trains on synthetic data).
Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Investor | Real Estate Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3550;
        border-radius: 12px;
        padding: 16px !important;
    }
    .pred-box {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border: 1px solid #40916c;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 8px 0;
    }
    .pred-box h1 { color: #95d5b2; font-size: 2.5rem; margin: 0; }
    .pred-box p  { color: #b7e4c7; margin: 4px 0 0 0; font-size: 1rem; }
    .section-header {
        color: #a5b4fc;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        border-bottom: 1px solid #2e3550;
        padding-bottom: 6px;
    }
    [data-testid="stSidebar"] {
        background: #131720 !important;
        border-right: 1px solid #2e3550;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────
LOCATIONS = [
    "Whitefield", "Electronic City", "Sarjapur Road", "HSR Layout",
    "Koramangala", "Indiranagar", "Marathahalli", "Bannerghatta Road",
    "Hebbal", "JP Nagar", "Bellandur", "Yelahanka", "Hennur",
    "Rajajinagar", "Malleshwaram", "Jayanagar", "BTM Layout",
    "Bommanahalli", "KR Puram", "Domlur", "Other"
]

LOCATION_MULTIPLIER = {
    "Koramangala": 1.6, "Indiranagar": 1.55, "Domlur": 1.45,
    "Malleshwaram": 1.4, "Jayanagar": 1.38, "HSR Layout": 1.35,
    "BTM Layout": 1.25, "JP Nagar": 1.2, "Whitefield": 1.18,
    "Sarjapur Road": 1.15, "Bellandur": 1.12, "Hebbal": 1.1,
    "Electronic City": 1.0, "Marathahalli": 1.05, "Bannerghatta Road": 1.0,
    "Rajajinagar": 1.1, "Yelahanka": 0.95, "Hennur": 0.92,
    "Bommanahalli": 0.9, "KR Puram": 0.88, "Other": 0.85,
}

def generate_synthetic_data(n=4000, seed=42):
    rng       = np.random.default_rng(seed)
    locations = rng.choice(LOCATIONS, size=n, p=[1/len(LOCATIONS)] * len(LOCATIONS))
    bhk       = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.35, 0.35, 0.15, 0.05])
    sqft      = np.clip(bhk * rng.uniform(450, 700, size=n) + rng.normal(0, 80, size=n), 300, 8000)
    bath      = np.clip(bhk + rng.choice([-1, 0, 1], size=n, p=[0.1, 0.7, 0.2]), 1, 6).astype(int)
    multiplier = np.array([LOCATION_MULTIPLIER[loc] for loc in locations])
    price_lakh = (4500 * sqft * multiplier * rng.uniform(0.85, 1.15, size=n)) / 100000
    return pd.DataFrame({
        "location": locations, "total_sqft": sqft.round(0),
        "bath": bath, "bhk": bhk, "price": price_lakh.round(2),
    })


# ─────────────────────────────────────────────
# TRAIN / LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training models on Bangalore housing data...")
def train_models():
    # Try pre-trained models first (for local use after running train_models.py)
    if os.path.exists("models/metrics.json"):
        try:
            with open("models/metrics.json") as f: metrics = json.load(f)
            with open("models/feature_columns.pkl", "rb") as f: feature_cols = pickle.load(f)
            with open("models/top_locations.pkl",   "rb") as f: top_locations = pickle.load(f)
            model_files = {
                "Linear Regression": "models/linear_regression.pkl",
                "Ridge Regression":  "models/ridge_regression.pkl",
                "Lasso Regression":  "models/lasso_regression.pkl",
                "Random Forest":     "models/random_forest.pkl",
            }
            models = {name: pickle.load(open(path, "rb")) for name, path in model_files.items()}
            return models, metrics, feature_cols, top_locations, "real"
        except Exception:
            pass

    # Otherwise train on synthetic data (Streamlit Cloud path)
    df            = generate_synthetic_data()
    top_locations = LOCATIONS
    X             = pd.get_dummies(df.drop(columns=["price"]), columns=["location"])
    y             = df["price"]
    feature_cols  = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10),
        "Lasso Regression":  Lasso(alpha=0.1, max_iter=10000),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
    }
    trained, metrics = {}, {}
    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        preds  = model.predict(X_test)
        cv_r2  = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        metrics[name] = {
            "MSE":   round(mean_squared_error(y_test, preds), 4),
            "RMSE":  round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "MAE":   round(mean_absolute_error(y_test, preds), 4),
            "R2":    round(r2_score(y_test, preds), 4),
            "CV_R2": round(cv_r2, 4),
        }
        trained[name] = model

    return trained, metrics, feature_cols, top_locations, "synthetic"


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_price(model, location, sqft, bath, bhk, feature_cols):
    row = {col: 0 for col in feature_cols}
    row.update({"total_sqft": sqft, "bath": bath, "bhk": bhk})
    loc_col = f"location_{location}"
    if loc_col in row:
        row[loc_col] = 1
    return round(max(model.predict(pd.DataFrame([row]))[0], 1), 2)


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def plot_metrics(metrics):
    names       = list(metrics.keys())
    short       = [n.replace(" Regression", "").replace(" ", "\n") for n in names]
    r2s         = [metrics[m]["R2"]   for m in names]
    rmses       = [metrics[m]["RMSE"] for m in names]
    colors      = ["#6366f1", "#8b5cf6", "#a78bfa", "#22d3ee"]
    best        = r2s.index(max(r2s))
    bar_colors  = ["#f59e0b" if i == best else c for i, c in enumerate(colors)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#131720"); ax.spines[:].set_color("#2e3550"); ax.tick_params(colors="#9ca3af")

    bars = axes[0].bar(short, r2s, color=bar_colors, edgecolor="#2e3550", linewidth=0.8)
    axes[0].set_title("R² Score (Higher = Better)", color="#a5b4fc", fontsize=12, pad=10)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("R²", color="#9ca3af")
    for bar, val in zip(bars, r2s):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    bars2 = axes[1].bar(short, rmses, color=colors, edgecolor="#2e3550", linewidth=0.8)
    axes[1].set_title("RMSE (Lower = Better)", color="#a5b4fc", fontsize=12, pad=10)
    axes[1].set_ylabel("RMSE (Lakhs)", color="#9ca3af")
    for bar, val in zip(bars2, rmses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    fig.patch.set_alpha(0); plt.tight_layout()
    return fig


def plot_radar(metrics):
    names   = list(metrics.keys())
    colors  = ["#6366f1", "#8b5cf6", "#a78bfa", "#22d3ee"]
    cats    = ["R²", "1/RMSE", "1/MAE", "CV R²"]

    def normalize(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 0 else arr

    data = np.column_stack([
        normalize(np.array([metrics[m]["R2"]    for m in names])),
        normalize(1 / (np.array([metrics[m]["RMSE"] for m in names]) + 1e-6)),
        normalize(1 / (np.array([metrics[m]["MAE"]  for m in names]) + 1e-6)),
        normalize(np.array([metrics[m]["CV_R2"] for m in names])),
    ])
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True}, facecolor="#0e1117")
    ax.set_facecolor("#131720"); ax.spines["polar"].set_color("#2e3550")
    for i, (name, row) in enumerate(zip(names, data)):
        vals = row.tolist() + row[:1].tolist()
        ax.plot(angles, vals, color=colors[i], linewidth=2, label=name.replace(" Regression", ""))
        ax.fill(angles, vals, color=colors[i], alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, color="#9ca3af", fontsize=10)
    ax.set_yticks([]); ax.grid(color="#2e3550", linewidth=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), labelcolor="white", framealpha=0, fontsize=8)
    ax.set_title("Model Comparison Radar", color="#a5b4fc", pad=20, fontsize=11)
    fig.patch.set_alpha(0)
    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    models, metrics, feature_cols, top_locations, data_source = train_models()

    st.markdown("""
    <div style="text-align:center; padding: 24px 0 8px 0;">
        <h1 style="color:#a5b4fc; font-size:2.4rem; margin:0;">🏠 Smart Investor</h1>
        <p style="color:#6b7280; font-size:1.05rem; margin:4px 0 0 0;">
            Bangalore Real Estate Price Predictor · Multi-Model Comparison
        </p>
    </div>
    """, unsafe_allow_html=True)

    if data_source == "synthetic":
        st.info("📊 Running on **synthetic training data** (realistic Bangalore market patterns). For production accuracy, train locally with the Kaggle dataset and commit the `models/` folder to your repo.", icon="ℹ️")

    st.markdown("---")

    with st.sidebar:
        st.markdown("### 🔍 Property Details")
        st.markdown("<hr style='border-color:#2e3550;margin:8px 0 16px 0'>", unsafe_allow_html=True)
        location = st.selectbox("📍 Location", options=sorted(top_locations))
        sqft     = st.slider("📐 Total Area (sq ft)", 300, 10000, 1200, 50)
        bhk      = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5, 6], index=1)
        bath     = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5, 6], index=1)
        selected = st.radio("🤖 Primary Model", list(models.keys()),
                            index=list(models.keys()).index("Random Forest"))

    col_pred, col_charts = st.columns([1, 2], gap="large")

    with col_pred:
        st.markdown('<div class="section-header">💰 Price Predictions</div>', unsafe_allow_html=True)
        all_preds = {n: predict_price(m, location, sqft, bath, bhk, feature_cols) for n, m in models.items()}
        price     = all_preds[selected]

        st.markdown(f"""
        <div class="pred-box">
            <p>Estimated Price ({selected})</p>
            <h1>₹ {price:.2f} L</h1>
            <p>≈ ₹ {price/100:.2f} Cr &nbsp;|&nbsp; ₹ {int(price*100000/sqft):,}/sqft</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">All Model Predictions</div>', unsafe_allow_html=True)
        best_model = max(metrics, key=lambda k: metrics[k]["R2"])
        for name, p in all_preds.items():
            c1, c2 = st.columns([3, 2])
            with c1: st.markdown(f"**{'⭐ ' if name == best_model else ''}{name}**")
            with c2: st.markdown(f"**₹ {p:.2f} L**")
            st.progress(min(p / max(all_preds.values()), 1.0))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Property Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Detail": ["Location", "Area", "BHK", "Bathrooms", "Price/sqft"],
            "Value":  [location, f"{sqft:,} sq ft", f"{bhk} BHK", str(bath), f"₹ {int(price*100000/sqft):,}"]
        }), hide_index=True, use_container_width=True)

    with col_charts:
        st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
        st.pyplot(plot_metrics(metrics), use_container_width=True)
        col_r, col_t = st.columns(2)
        with col_r:
            st.pyplot(plot_radar(metrics), use_container_width=True)
        with col_t:
            st.markdown('<div class="section-header">📈 Metrics Table</div>', unsafe_allow_html=True)
            rows = [{"Model": ("⭐ " if n == best_model else "") + n.replace(" Regression", ""),
                     "R²": metrics[n]["R2"], "RMSE": metrics[n]["RMSE"],
                     "MAE": metrics[n]["MAE"], "CV R²": metrics[n]["CV_R2"]} for n in metrics]
            st.dataframe(
                pd.DataFrame(rows).style
                    .highlight_max(subset=["R²", "CV R²"], color="#1a472a")
                    .highlight_min(subset=["RMSE", "MAE"], color="#1a472a"),
                hide_index=True, use_container_width=True, height=210)
            st.caption("⭐ = Best model by R²")

    st.markdown("---")
    st.markdown('<div style="text-align:center;color:#4b5563;font-size:0.85rem;padding:8px 0">Smart Investor · Scikit-learn + Streamlit · Bangalore Housing</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()