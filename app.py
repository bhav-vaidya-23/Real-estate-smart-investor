"""
Smart Investor — Real Estate Price Predictor Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Investor | Real Estate Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3550;
        border-radius: 12px;
        padding: 16px !important;
    }

    /* Prediction box */
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

    /* Winner badge */
    .winner-badge {
        background: linear-gradient(135deg, #7b2d8b, #a855f7);
        border-radius: 8px;
        padding: 4px 12px;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
    }

    /* Section headers */
    .section-header {
        color: #a5b4fc;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        border-bottom: 1px solid #2e3550;
        padding-bottom: 6px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #131720 !important;
        border-right: 1px solid #2e3550;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all trained models and metadata."""
    if not os.path.exists("models/metrics.json"):
        return None, None, None, None

    with open("models/metrics.json") as f:
        metrics = json.load(f)

    with open("models/feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    with open("models/top_locations.pkl", "rb") as f:
        top_locations = pickle.load(f)

    model_files = {
        "Linear Regression":  "models/linear_regression.pkl",
        "Ridge Regression":   "models/ridge_regression.pkl",
        "Lasso Regression":   "models/lasso_regression.pkl",
        "Random Forest":      "models/random_forest.pkl",
    }
    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    return models, metrics, feature_cols, top_locations


# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_price(model, location, sqft, bath, bhk, feature_cols):
    """Build input vector and predict."""
    input_data = {col: 0 for col in feature_cols}
    input_data["total_sqft"] = sqft
    input_data["bath"] = bath
    input_data["bhk"] = bhk

    loc_col = f"location_{location}"
    if loc_col in input_data:
        input_data[loc_col] = 1

    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    return round(max(pred, 1), 2)  # ensure non-negative


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def plot_metrics(metrics):
    model_names = list(metrics.keys())
    short_names = [n.replace(" Regression", "").replace(" ", "\n") for n in model_names]
    r2_scores   = [metrics[m]["R2"]   for m in model_names]
    rmse_scores = [metrics[m]["RMSE"] for m in model_names]

    colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#22d3ee"]
    best_r2_idx = r2_scores.index(max(r2_scores))
    bar_colors = [("#f59e0b" if i == best_r2_idx else c) for i, c in enumerate(colors)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0e1117")

    for ax in axes:
        ax.set_facecolor("#131720")
        ax.spines[:].set_color("#2e3550")
        ax.tick_params(colors="#9ca3af")

    # R² bar chart
    bars = axes[0].bar(short_names, r2_scores, color=bar_colors, edgecolor="#2e3550", linewidth=0.8)
    axes[0].set_title("R² Score (Higher = Better)", color="#a5b4fc", fontsize=12, pad=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R²", color="#9ca3af")
    for bar, val in zip(bars, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    # RMSE bar chart
    bars2 = axes[1].bar(short_names, rmse_scores, color=colors, edgecolor="#2e3550", linewidth=0.8)
    axes[1].set_title("RMSE (Lower = Better)", color="#a5b4fc", fontsize=12, pad=10)
    axes[1].set_ylabel("RMSE (Lakhs)", color="#9ca3af")
    for bar, val in zip(bars2, rmse_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def plot_radar(metrics):
    """Radar chart comparing all models across metrics."""
    categories = ["R²", "1/RMSE", "1/MAE", "CV R²"]
    model_names = list(metrics.keys())
    colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#22d3ee"]

    # Normalize metrics 0-1
    r2s   = np.array([metrics[m]["R2"]    for m in model_names])
    rmses = np.array([1 / (metrics[m]["RMSE"] + 1e-6) for m in model_names])
    maes  = np.array([1 / (metrics[m]["MAE"]  + 1e-6) for m in model_names])
    cv_r2 = np.array([metrics[m]["CV_R2"] for m in model_names])

    def normalize(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else arr

    data_matrix = np.column_stack([
        normalize(r2s), normalize(rmses), normalize(maes), normalize(cv_r2)
    ])

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True}, facecolor="#0e1117")
    ax.set_facecolor("#131720")
    ax.spines["polar"].set_color("#2e3550")

    for i, (name, row) in enumerate(zip(model_names, data_matrix)):
        values = row.tolist() + row[:1].tolist()
        ax.plot(angles, values, color=colors[i], linewidth=2, label=name.replace(" Regression", ""))
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#9ca3af", fontsize=10)
    ax.set_yticks([])
    ax.grid(color="#2e3550", linewidth=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              labelcolor="white", framealpha=0, fontsize=8)
    ax.set_title("Model Comparison Radar", color="#a5b4fc", pad=20, fontsize=11)

    fig.patch.set_alpha(0)
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    models, metrics, feature_cols, top_locations = load_models()

    # ── HEADER ──
    st.markdown("""
    <div style="text-align:center; padding: 24px 0 8px 0;">
        <h1 style="color:#a5b4fc; font-size:2.4rem; margin:0;">🏠 Smart Investor</h1>
        <p style="color:#6b7280; font-size:1.05rem; margin:4px 0 0 0;">
            Bangalore Real Estate Price Predictor · Multi-Model Comparison
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── GUARD: models not trained yet ──
    if models is None:
        st.warning("⚠️ No trained models found. Please run `python train_models.py` first!")
        st.code("python train_models.py", language="bash")
        return

    # ─────────────────────────────────────────
    # SIDEBAR — INPUT FORM
    # ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Property Details")
        st.markdown("<hr style='border-color:#2e3550;margin:8px 0 16px 0'>", unsafe_allow_html=True)

        location = st.selectbox(
            "📍 Location",
            options=sorted(top_locations) + ["Other"],
            index=0,
            help="Select the property location in Bangalore"
        )

        sqft = st.slider(
            "📐 Total Area (sq ft)",
            min_value=300, max_value=10000, value=1200, step=50
        )

        bhk = st.selectbox(
            "🛏️ BHK (Bedrooms)",
            options=[1, 2, 3, 4, 5, 6],
            index=1
        )

        bath = st.selectbox(
            "🚿 Bathrooms",
            options=[1, 2, 3, 4, 5, 6],
            index=1
        )

        selected_model = st.radio(
            "🤖 Primary Model",
            options=list(models.keys()),
            index=list(models.keys()).index("Random Forest") if "Random Forest" in models else 0,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Price", use_container_width=True, type="primary")

    # ─────────────────────────────────────────
    # MAIN CONTENT
    # ─────────────────────────────────────────
    col_pred, col_charts = st.columns([1, 2], gap="large")

    with col_pred:
        st.markdown('<div class="section-header">💰 Price Predictions</div>', unsafe_allow_html=True)

        if predict_btn or True:  # Always show predictions
            all_preds = {}
            for mname, mobj in models.items():
                price = predict_price(mobj, location, sqft, bath, bhk, feature_cols)
                all_preds[mname] = price

            # Primary model highlighted
            primary_price = all_preds[selected_model]
            st.markdown(f"""
            <div class="pred-box">
                <p>Estimated Price ({selected_model})</p>
                <h1>₹ {primary_price:.2f} L</h1>
                <p>≈ ₹ {primary_price/100:.2f} Cr &nbsp;|&nbsp; ₹ {int(primary_price*100000/sqft):,}/sqft</p>
            </div>
            """, unsafe_allow_html=True)

            # All model predictions
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">All Model Predictions</div>', unsafe_allow_html=True)

            best_model = max(metrics, key=lambda k: metrics[k]["R2"])
            for mname, price in all_preds.items():
                badge = "⭐" if mname == best_model else ""
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    st.markdown(f"**{badge} {mname}**")
                with col_b:
                    st.markdown(f"**₹ {price:.2f} L**")
                st.progress(min(price / max(all_preds.values()), 1.0))

            # Property summary card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Property Summary</div>', unsafe_allow_html=True)
            info_df = pd.DataFrame({
                "Detail": ["Location", "Area", "BHK", "Bathrooms", "Price/sqft"],
                "Value": [
                    location, f"{sqft:,} sq ft", f"{bhk} BHK",
                    str(bath), f"₹ {int(primary_price*100000/sqft):,}"
                ]
            })
            st.dataframe(info_df, hide_index=True, use_container_width=True)

    with col_charts:
        st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
        st.pyplot(plot_metrics(metrics), use_container_width=True)

        col_r, col_t = st.columns([1, 1])
        with col_r:
            st.pyplot(plot_radar(metrics), use_container_width=True)

        with col_t:
            st.markdown('<div class="section-header">📈 Metrics Table</div>', unsafe_allow_html=True)
            rows = []
            best_r2 = max(metrics, key=lambda k: metrics[k]["R2"])
            for mname, m in metrics.items():
                rows.append({
                    "Model":  ("⭐ " if mname == best_r2 else "   ") + mname.replace(" Regression", ""),
                    "R²":     m["R2"],
                    "RMSE":   m["RMSE"],
                    "MAE":    m["MAE"],
                    "CV R²":  m["CV_R2"],
                })
            metrics_df = pd.DataFrame(rows)
            st.dataframe(
                metrics_df.style.highlight_max(subset=["R²", "CV R²"], color="#1a472a")
                                .highlight_min(subset=["RMSE", "MAE"], color="#1a472a"),
                hide_index=True, use_container_width=True, height=210
            )
            st.caption("⭐ = Best overall model by R²")

    # ── FOOTER ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#4b5563; font-size:0.85rem; padding:8px 0">
        Smart Investor · Built with Scikit-learn + Streamlit · Bangalore Housing Dataset
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
