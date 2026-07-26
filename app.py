"""
📊 YouTube Video Performance Analysis — Streamlit App
=====================================================
A production-ready Streamlit dashboard that ingests a YouTube channel
analytics CSV, runs EDA, trains a Random Forest Regressor to predict
Estimated Revenue, and surfaces actionable insights.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────── page config ────────────────────────────
st.set_page_config(
    page_title="YouTube Video Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── custom CSS ─────────────────────────────
st.markdown(
    """
    <style>
    /* ─── Root variables ─── */
    :root {
        --primary: #FF0000;
        --primary-dark: #CC0000;
        --accent: #1DB954;
        --bg-dark: #0E1117;
        --card-bg: rgba(30, 33, 43, 0.65);
        --card-border: rgba(255, 255, 255, 0.06);
        --text-primary: #FAFAFA;
        --text-muted: #8B949E;
        --glass-blur: blur(14px);
        --radius: 14px;
    }

    /* ─── Global tweaks ─── */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #161B22 50%, #1A1F29 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B22 0%, #0E1117 100%);
        border-right: 1px solid var(--card-border);
    }

    /* ─── Header banner ─── */
    .hero-banner {
        background: linear-gradient(135deg, #FF0000 0%, #FF4444 40%, #FF6B6B 100%);
        border-radius: var(--radius);
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(255, 0, 0, 0.18);
        text-align: center;
    }
    .hero-banner h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0;
    }

    /* ─── Glass metric cards ─── */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--card-border);
        border-radius: var(--radius);
        padding: 1.3rem 1.5rem;
        text-align: center;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
    }
    .metric-label {
        font-size: 0.82rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ─── Section headers ─── */
    .section-header {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-primary);
        border-left: 4px solid var(--primary);
        padding-left: 0.75rem;
        margin: 2rem 0 1rem 0;
    }

    /* ─── Hide default Streamlit branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────── helper funcs ───────────────────────────

def render_metric(label: str, value: str) -> str:
    """Return HTML for a glass-style metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def section(title: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ──────────────────────────── sidebar ────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded_file = st.file_uploader(
        "Upload YouTube Analytics CSV",
        type=["csv"],
        help="Upload the YouTube channel performance analytics CSV file.",
    )

    st.markdown("---")
    st.markdown(
        "**Model Parameters**",
    )
    n_estimators = st.slider("Number of Trees", 50, 500, 100, step=50)
    test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)
    random_state = st.number_input("Random State", value=42, step=1)

    st.markdown("---")
    st.markdown(
        "<small style='color:#8B949E;'>Built with Streamlit · ML powered by scikit-learn</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────── main area ──────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>📊 YouTube Video Performance Analytics</h1>
        <p>Upload your channel data · Explore insights · Predict revenue with ML</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("👈 **Upload a CSV file** from the sidebar to get started.")
    st.stop()

# ─────────────────── Step 1 — Load & Preview ─────────────────────────
data = pd.read_csv(uploaded_file)

section("📁 Dataset Preview")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.markdown(render_metric("Rows", f"{data.shape[0]:,}"), unsafe_allow_html=True)
with col_info2:
    st.markdown(render_metric("Columns", f"{data.shape[1]:,}"), unsafe_allow_html=True)
with col_info3:
    st.markdown(render_metric("Numeric Cols", f"{data.select_dtypes(include=[np.number]).shape[1]}"), unsafe_allow_html=True)
with col_info4:
    missing = int(data.isnull().sum().sum())
    st.markdown(render_metric("Missing Values", f"{missing:,}"), unsafe_allow_html=True)

st.dataframe(data.head(20), use_container_width=True, height=320)

with st.expander("📋 Column Types & Null Counts"):
    info_df = pd.DataFrame(
        {
            "Type": data.dtypes.astype(str),
            "Non-Null": data.notnull().sum(),
            "Null": data.isnull().sum(),
        }
    )
    st.dataframe(info_df, use_container_width=True)


# ─────────────────── Step 2 — Data Cleaning ──────────────────────────
section("🧹 Data Cleaning")

rows_before = len(data)
data = data.dropna()
rows_after = len(data)
dropped = rows_before - rows_after

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(render_metric("Before Cleaning", f"{rows_before:,}"), unsafe_allow_html=True)
with c2:
    st.markdown(render_metric("After Cleaning", f"{rows_after:,}"), unsafe_allow_html=True)
with c3:
    st.markdown(render_metric("Rows Dropped", f"{dropped:,}"), unsafe_allow_html=True)

if dropped > 0:
    st.warning(f"⚠️ {dropped} row(s) with missing values were removed.")
else:
    st.success("✅ No missing values found — dataset is clean!")


# ─────────────────── Step 3 — EDA ────────────────────────────────────
section("📊 Exploratory Data Analysis")

# --- 3a: Revenue Distribution + Top 10 Videos + Views vs Revenue + Heatmap ---
tab_dist, tab_top, tab_scatter, tab_heatmap = st.tabs(
    ["📈 Revenue Distribution", "🏆 Top 10 Videos", "🔍 Views vs Revenue", "🔥 Correlation Heatmap"]
)

with tab_dist:
    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
    ax_dist.hist(data["Estimated Revenue (USD)"], bins=30, color="#00C9A7", edgecolor="#0E1117", alpha=0.88)
    ax_dist.set_title("Revenue Distribution", fontsize=14, fontweight="bold", color="white")
    ax_dist.set_xlabel("Estimated Revenue (USD)", color="white")
    ax_dist.set_ylabel("Frequency", color="white")
    ax_dist.set_facecolor("#161B22")
    fig_dist.patch.set_facecolor("#0E1117")
    ax_dist.tick_params(colors="white")
    ax_dist.spines["bottom"].set_color("#30363D")
    ax_dist.spines["left"].set_color("#30363D")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    st.pyplot(fig_dist)

with tab_top:
    top_videos = data.nlargest(10, "Estimated Revenue (USD)")[["ID", "Estimated Revenue (USD)"]]
    fig_top, ax_top = plt.subplots(figsize=(10, 5))
    bars = ax_top.barh(
        top_videos["ID"].astype(str),
        top_videos["Estimated Revenue (USD)"],
        color=sns.color_palette("magma", 10),
        edgecolor="#0E1117",
    )
    ax_top.set_title("Top 10 Videos by Revenue", fontsize=14, fontweight="bold", color="white")
    ax_top.set_xlabel("Revenue (USD)", color="white")
    ax_top.set_ylabel("Video ID", color="white")
    ax_top.set_facecolor("#161B22")
    fig_top.patch.set_facecolor("#0E1117")
    ax_top.tick_params(colors="white")
    ax_top.spines["bottom"].set_color("#30363D")
    ax_top.spines["left"].set_color("#30363D")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    st.pyplot(fig_top)

with tab_scatter:
    fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
    ax_sc.scatter(
        data["Views"],
        data["Estimated Revenue (USD)"],
        alpha=0.55,
        c=data["Estimated Revenue (USD)"],
        cmap="cool",
        edgecolors="#0E1117",
        s=50,
    )
    ax_sc.set_title("Views vs Revenue", fontsize=14, fontweight="bold", color="white")
    ax_sc.set_xlabel("Views", color="white")
    ax_sc.set_ylabel("Revenue (USD)", color="white")
    ax_sc.set_facecolor("#161B22")
    fig_sc.patch.set_facecolor("#0E1117")
    ax_sc.tick_params(colors="white")
    ax_sc.spines["bottom"].set_color("#30363D")
    ax_sc.spines["left"].set_color("#30363D")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    st.pyplot(fig_sc)

with tab_heatmap:
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    top_corr_cols = corr["Estimated Revenue (USD)"].abs().sort_values(ascending=False).head(10).index
    fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data[top_corr_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax_hm,
        linewidths=0.5,
        linecolor="#0E1117",
        cbar_kws={"shrink": 0.8},
    )
    ax_hm.set_title("Correlation Heatmap (Top Features)", fontsize=14, fontweight="bold", color="white")
    ax_hm.set_facecolor("#161B22")
    fig_hm.patch.set_facecolor("#0E1117")
    ax_hm.tick_params(colors="white")
    st.pyplot(fig_hm)


# ─────────────────── Step 4 — Feature Engineering ────────────────────
section("⚙️ Feature Engineering")

data["Revenue per View"] = data["Estimated Revenue (USD)"] / data["Views"]
data["Engagement Rate"] = (
    (data["Likes"] + data["Shares"] + data["New Comments"]) / data["Views"] * 100
)

# Handle potential inf/NaN from division by zero
data["Revenue per View"] = data["Revenue per View"].replace([np.inf, -np.inf], 0).fillna(0)
data["Engagement Rate"] = data["Engagement Rate"].replace([np.inf, -np.inf], 0).fillna(0)

new_feat_df = data[["Revenue per View", "Engagement Rate"]].describe().T
st.dataframe(new_feat_df.style.format("{:.6f}"), use_container_width=True)
st.success("✅ 2 new features created: *Revenue per View*, *Engagement Rate*")


# ─────────────────── Step 5 — Feature Selection ──────────────────────
section("🎯 Feature Selection")

selected_features = [
    "Views",
    "Subscribers",
    "Likes",
    "Shares",
    "New Comments",
    "Engagement Rate",
]

target = "Estimated Revenue (USD)"

# Verify columns exist before proceeding
missing_cols = [c for c in selected_features + [target] if c not in data.columns]
if missing_cols:
    st.error(f"❌ The following required columns are missing from the dataset: {missing_cols}")
    st.stop()

X = data[selected_features]
y = data[target]

fc1, fc2 = st.columns(2)
with fc1:
    st.markdown(render_metric("Features", f"{X.shape[1]}"), unsafe_allow_html=True)
with fc2:
    st.markdown(render_metric("Samples", f"{X.shape[0]:,}"), unsafe_allow_html=True)

st.markdown("**Selected features:**")
st.code(", ".join(selected_features), language="text")


# ─────────────────── Step 6 — Model Training ─────────────────────────
section("🤖 Model Training — Random Forest Regressor")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

with st.spinner("🔄 Training model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(render_metric("R² Score", f"{r2:.4f}"), unsafe_allow_html=True)
with m2:
    st.markdown(render_metric("RMSE", f"{rmse:.4f}"), unsafe_allow_html=True)
with m3:
    st.markdown(render_metric("MAE", f"{mae:.4f}"), unsafe_allow_html=True)

# --- Prediction vs Actual ---
tab_pred, tab_imp = st.tabs(["📉 Prediction vs Actual", "📊 Feature Importance"])

with tab_pred:
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    ax_pred.scatter(y_test, y_pred, alpha=0.6, c="#1DB954", edgecolors="#0E1117", s=55)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax_pred.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax_pred.set_title("Prediction vs Actual Revenue", fontsize=14, fontweight="bold", color="white")
    ax_pred.set_xlabel("Actual Revenue (USD)", color="white")
    ax_pred.set_ylabel("Predicted Revenue (USD)", color="white")
    ax_pred.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white")
    ax_pred.set_facecolor("#161B22")
    fig_pred.patch.set_facecolor("#0E1117")
    ax_pred.tick_params(colors="white")
    ax_pred.spines["bottom"].set_color("#30363D")
    ax_pred.spines["left"].set_color("#30363D")
    ax_pred.spines["top"].set_visible(False)
    ax_pred.spines["right"].set_visible(False)
    st.pyplot(fig_pred)

with tab_imp:
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=selected_features).sort_values(ascending=True)
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    feat_imp.plot(
        kind="barh",
        color=sns.color_palette("viridis", len(feat_imp)),
        edgecolor="#0E1117",
        ax=ax_imp,
    )
    ax_imp.set_title("Feature Importance", fontsize=14, fontweight="bold", color="white")
    ax_imp.set_xlabel("Importance", color="white")
    ax_imp.set_facecolor("#161B22")
    fig_imp.patch.set_facecolor("#0E1117")
    ax_imp.tick_params(colors="white")
    ax_imp.spines["bottom"].set_color("#30363D")
    ax_imp.spines["left"].set_color("#30363D")
    ax_imp.spines["top"].set_visible(False)
    ax_imp.spines["right"].set_visible(False)
    st.pyplot(fig_imp)


# ─────────────────── Step 7 — Revenue Insights ───────────────────────
section("💡 Revenue Insights")

tab_monthly, tab_daily, tab_box, tab_likes = st.tabs(
    ["📅 Monthly Revenue", "📆 Revenue by Day", "📦 Monthly Boxplot", "❤️ Revenue vs Likes"]
)

with tab_monthly:
    monthly_revenue = data.groupby("Month")["Estimated Revenue (USD)"].sum()
    fig_mo, ax_mo = plt.subplots(figsize=(10, 5))
    monthly_revenue.plot(kind="bar", color="#00C9A7", edgecolor="#0E1117", ax=ax_mo)
    ax_mo.set_title("Total Revenue by Month", fontsize=14, fontweight="bold", color="white")
    ax_mo.set_xlabel("Month", color="white")
    ax_mo.set_ylabel("Revenue (USD)", color="white")
    ax_mo.set_facecolor("#161B22")
    fig_mo.patch.set_facecolor("#0E1117")
    ax_mo.tick_params(colors="white")
    ax_mo.spines["bottom"].set_color("#30363D")
    ax_mo.spines["left"].set_color("#30363D")
    ax_mo.spines["top"].set_visible(False)
    ax_mo.spines["right"].set_visible(False)
    st.pyplot(fig_mo)

with tab_daily:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if "Day of Week" in data.columns:
        day_revenue = data.groupby("Day of Week")["Estimated Revenue (USD)"].sum().reindex(day_order)
        fig_day, ax_day = plt.subplots(figsize=(10, 5))
        day_revenue.plot(kind="bar", color="#FF6B6B", edgecolor="#0E1117", ax=ax_day)
        ax_day.set_title("Revenue by Day of Week", fontsize=14, fontweight="bold", color="white")
        ax_day.set_xlabel("Day of Week", color="white")
        ax_day.set_ylabel("Revenue (USD)", color="white")
        ax_day.set_facecolor("#161B22")
        fig_day.patch.set_facecolor("#0E1117")
        ax_day.tick_params(colors="white", axis="both")
        ax_day.spines["bottom"].set_color("#30363D")
        ax_day.spines["left"].set_color("#30363D")
        ax_day.spines["top"].set_visible(False)
        ax_day.spines["right"].set_visible(False)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_day)
    else:
        st.warning("Column 'Day of Week' not found in dataset.")

with tab_box:
    fig_box, ax_box = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        x="Month",
        y="Estimated Revenue (USD)",
        data=data,
        ax=ax_box,
        palette="Set2",
        flierprops=dict(marker="o", markerfacecolor="#FF4444", markersize=5, linestyle="none"),
    )
    ax_box.set_title("Revenue Distribution by Month", fontsize=14, fontweight="bold", color="white")
    ax_box.set_facecolor("#161B22")
    fig_box.patch.set_facecolor("#0E1117")
    ax_box.tick_params(colors="white")
    ax_box.set_xlabel("Month", color="white")
    ax_box.set_ylabel("Estimated Revenue (USD)", color="white")
    ax_box.spines["bottom"].set_color("#30363D")
    ax_box.spines["left"].set_color("#30363D")
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    st.pyplot(fig_box)

with tab_likes:
    fig_likes, ax_likes = plt.subplots(figsize=(10, 5))
    sns.regplot(
        x="Likes",
        y="Estimated Revenue (USD)",
        data=data,
        ax=ax_likes,
        scatter_kws={"alpha": 0.5, "color": "#9B59B6", "edgecolors": "#0E1117", "s": 45},
        line_kws={"color": "#FF4444", "linewidth": 2},
    )
    ax_likes.set_title("Revenue vs Likes (with Regression)", fontsize=14, fontweight="bold", color="white")
    ax_likes.set_facecolor("#161B22")
    fig_likes.patch.set_facecolor("#0E1117")
    ax_likes.tick_params(colors="white")
    ax_likes.set_xlabel("Likes", color="white")
    ax_likes.set_ylabel("Estimated Revenue (USD)", color="white")
    ax_likes.spines["bottom"].set_color("#30363D")
    ax_likes.spines["left"].set_color("#30363D")
    ax_likes.spines["top"].set_visible(False)
    ax_likes.spines["right"].set_visible(False)
    st.pyplot(fig_likes)

# ─────────────────── Step 8 — Download Predictions ───────────────────
section("📥 Download Predictions")

results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
results_df["Abs Error"] = results_df["Error"].abs()
results_df.index = y_test.index
results_df.index.name = "Sample Index"

st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True, height=300)

csv_results = results_df.to_csv().encode("utf-8")
st.download_button(
    label="⬇️ Download Predictions CSV",
    data=csv_results,
    file_name="youtube_revenue_predictions.csv",
    mime="text/csv",
)
