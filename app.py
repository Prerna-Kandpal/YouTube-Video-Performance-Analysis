"""
📊 YouTube Video Performance Analytics & Revenue Predictor — Streamlit App
===========================================================================
A production-ready, high-performance Streamlit application featuring a modern
glassmorphic UI, interactive EDA, automated feature engineering, Random Forest
revenue prediction, and an interactive real-time revenue simulator.

Usage:
    streamlit run app.py
"""

import os
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

# ──────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="YouTube Performance & Revenue Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ─────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Global Background ─── */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(255, 0, 0, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(0, 245, 160, 0.04) 0%, transparent 40%),
                    linear-gradient(135deg, #090C10 0%, #0D1117 50%, #161B22 100%);
        color: #E6EDF3;
    }

    /* ─── Sidebar Styling ─── */
    section[data-testid="stSidebar"] {
        background: #0D1117;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Outfit', sans-serif;
        color: #FF2E4D;
        font-size: 1.3rem;
        letter-spacing: -0.3px;
    }

    /* ─── Hero Header Banner ─── */
    .hero-container {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.85) 0%, rgba(204, 0, 0, 0.9) 50%, rgba(139, 0, 0, 0.95) 100%);
        border-radius: 20px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(255, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.08rem;
        font-weight: 400;
        margin: 0;
        max-width: 800px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.35rem 0.9rem;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #00F5A0;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ─── Glass Metric Cards ─── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2rem;
    }
    .glass-card {
        background: rgba(22, 27, 34, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 46, 77, 0.4);
        box-shadow: 0 14px 32px rgba(255, 46, 77, 0.15);
    }
    .glass-card-icon {
        font-size: 1.4rem;
        margin-bottom: 0.4rem;
    }
    .glass-card-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .glass-card-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        color: #F0F6FC;
        letter-spacing: -0.5px;
    }
    .glass-card-sub {
        font-size: 0.75rem;
        color: #00F5A0;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* ─── Section Titles ─── */
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #F0F6FC;
        margin: 2rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .section-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 22px;
        background: linear-gradient(180deg, #FF2E4D 0%, #FF6B6B 100%);
        border-radius: 4px;
    }

    /* ─── Custom Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(13, 17, 23, 0.8);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        color: #8B949E;
        font-weight: 500;
        font-size: 0.92rem;
        padding: 0 18px;
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF2E4D 0%, #CC0029 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(255, 46, 77, 0.3);
    }

    /* ─── Buttons & Controls ─── */
    .stButton>button {
        background: linear-gradient(135deg, #FF2E4D 0%, #CC0029 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(255, 46, 77, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 46, 77, 0.4);
        color: #FFFFFF;
    }

    /* ─── Dataframe & Tables ─── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ─── Hide default Streamlit elements ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────── Helper Functions ────────────────────────────

def render_glass_card(icon: str, label: str, value: str, subtext: str = ""):
    """Render a modern glassmorphic metric card."""
    sub_html = f'<div class="glass-card-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-card-icon">{icon}</div>
            <div class="glass-card-label">{label}</div>
            <div class="glass-card-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_matplotlib_dark():
    """Apply modern dark theme styling to matplotlib figures."""
    try:
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": "#0E1117",
                "axes.facecolor": "#161B22",
                "axes.edgecolor": "#30363D",
                "axes.labelcolor": "#8B949E",
                "axes.titlecolor": "#F0F6FC",
                "xtick.color": "#8B949E",
                "ytick.color": "#8B949E",
                "grid.color": "#21262D",
                "grid.linestyle": "--",
                "font.family": "sans-serif",
            }
        )
    except Exception:
        plt.style.use("dark_background")


setup_matplotlib_dark()


# ──────────────────────────── Sidebar & Data Input ────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ App Controls")
    
    # Mode selection: Use sample CSV vs Upload
    data_source_mode = st.radio(
        "Data Source Mode",
        ["Use Default Dataset", "Upload Custom CSV"],
        index=0,
        help="Select whether to use the included YouTube performance dataset or upload your own.",
    )

    uploaded_file = None
    if data_source_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload YouTube Analytics CSV",
            type=["csv"],
            help="Upload your YouTube channel performance CSV dataset.",
        )

    st.markdown("---")
    st.markdown("### 🤖 Model Hyperparameters")
    n_estimators = st.slider("Random Forest Trees", 50, 500, 100, step=50)
    test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)
    random_state = st.number_input("Random Seed State", value=42, step=1)

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #8B949E; font-size: 0.8rem;'>
            YouTube Performance Analytics v2.0<br/>
            Engineered with Streamlit & scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────── Dataset Loading ────────────────────────────

DEFAULT_CSV_PATH = "youtube_channel_data.csv"

@st.cache_data
def load_csv_data(source):
    if isinstance(source, str):
        if os.path.exists(source):
            return pd.read_csv(source)
        else:
            return None
    else:
        return pd.read_csv(source)

data_raw = None
if data_source_mode == "Upload Custom CSV":
    if uploaded_file is not None:
        data_raw = load_csv_data(uploaded_file)
    else:
        st.info("👈 Please upload a CSV dataset file using the sidebar to begin analysis.")
        st.stop()
else:
    # Use default CSV
    if os.path.exists(DEFAULT_CSV_PATH):
        data_raw = load_csv_data(DEFAULT_CSV_PATH)
    else:
        st.error(f"❌ Default dataset file `{DEFAULT_CSV_PATH}` not found. Please upload a CSV via the sidebar.")
        st.stop()

if data_raw is None or data_raw.empty:
    st.error("❌ Failed to load dataset. Please verify the CSV format.")
    st.stop()

data = data_raw.copy()


# ──────────────────────────── Data Cleaning ────────────────────────────

rows_before = len(data)
data = data.dropna()
rows_after = len(data)
rows_dropped = rows_before - rows_after


# ──────────────────────────── Feature Engineering ────────────────────────────

if "Views" in data.columns and "Estimated Revenue (USD)" in data.columns:
    data["Revenue per View"] = data["Estimated Revenue (USD)"] / data["Views"]
    data["Revenue per View"] = data["Revenue per View"].replace([np.inf, -np.inf], 0).fillna(0)

if all(col in data.columns for col in ["Likes", "Shares", "New Comments", "Views"]):
    data["Engagement Rate"] = (
        (data["Likes"] + data["Shares"] + data["New Comments"]) / data["Views"] * 100
    )
    data["Engagement Rate"] = data["Engagement Rate"].replace([np.inf, -np.inf], 0).fillna(0)


# ──────────────────────────── Model Training ────────────────────────────

selected_features = [
    "Views",
    "Subscribers",
    "Likes",
    "Shares",
    "New Comments",
    "Engagement Rate",
]

target_col = "Estimated Revenue (USD)"

# Validate columns
avail_features = [f for f in selected_features if f in data.columns]
has_target = target_col in data.columns

model_trained = False
r2, rmse, mae = 0.0, 0.0, 0.0
y_test, y_pred = None, None
feat_imp = None
rf_model = None

if len(avail_features) >= 3 and has_target:
    X = data[avail_features]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    feat_imp = pd.Series(rf_model.feature_importances_, index=avail_features).sort_values(ascending=True)
    model_trained = True


# ──────────────────────────── Main Dashboard Header ────────────────────────────

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">✨ Production Analytics Dashboard</div>
        <div class="hero-title">YouTube Video Performance & Revenue Engine</div>
        <div class="hero-subtitle">
            Comprehensive exploratory analysis, algorithmic feature engineering, and high-accuracy Random Forest Revenue forecasting.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────── Executive Metric Cards ────────────────────────────

tot_revenue = data["Estimated Revenue (USD)"].sum() if "Estimated Revenue (USD)" in data.columns else 0
tot_views = data["Views"].sum() if "Views" in data.columns else 0
avg_views = data["Views"].mean() if "Views" in data.columns else 0
avg_ctr = data["Video Thumbnail CTR (%)"].mean() if "Video Thumbnail CTR (%)" in data.columns else 0
avg_eng = data["Engagement Rate"].mean() if "Engagement Rate" in data.columns else 0
tot_subs = data["Subscribers"].sum() if "Subscribers" in data.columns else 0

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

with m_col1:
    render_glass_card("💰", "Total Revenue", f"${tot_revenue:,.2f}", "+ Real-time Sum")
with m_col2:
    render_glass_card("👁️", "Total Views", f"{tot_views:,.0f}", f"Avg: {avg_views:,.0f}/video")
with m_col3:
    render_glass_card("🎯", "Avg Thumbnail CTR", f"{avg_ctr:.2f}%", "Channel Average")
with m_col4:
    render_glass_card("⚡", "Avg Engagement", f"{avg_eng:.2f}%", "Likes + Comments + Shares")
with m_col5:
    render_glass_card("👥", "Total Subscribers", f"{tot_subs:,.0f}", "Gained Across Videos")
with m_col6:
    render_glass_card("🤖", "Model R² Score", f"{r2:.3f}" if model_trained else "N/A", "Accuracy Score")


# ──────────────────────────── Main Application Tabs ────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Executive EDA",
        "📆 Seasonal Trends",
        "⚙️ Feature Engineering",
        "🤖 ML Model Engine",
        "🔮 Revenue Predictor",
        "🗃️ Raw Data & Export",
    ]
)


# ─── TAB 1: Executive EDA ───
with tab1:
    st.markdown('<div class="section-title">Exploratory Data Analysis Overview</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 Revenue Distribution Across Videos")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(data["Estimated Revenue (USD)"], bins=35, kde=True, color="#FF2E4D", ax=ax, alpha=0.7)
        ax.set_title("Distribution of Estimated Revenue (USD)", pad=12, fontweight="bold")
        ax.set_xlabel("Revenue (USD)")
        ax.set_ylabel("Video Count")
        st.pyplot(fig)

    with col_b:
        st.markdown("#### 🏆 Top 10 Revenue Generating Videos")
        top_10 = data.nlargest(10, "Estimated Revenue (USD)")[["ID", "Estimated Revenue (USD)", "Views"]]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(top_10["ID"].astype(str), top_10["Estimated Revenue (USD)"], color="#00F5A0", alpha=0.85)
        ax.set_title("Top 10 Videos by Revenue (USD)", pad=12, fontweight="bold")
        ax.set_xlabel("Estimated Revenue (USD)")
        ax.set_ylabel("Video ID")
        st.pyplot(fig)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### 🔍 Views vs. Revenue Relationship")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(
            data=data,
            x="Views",
            y="Estimated Revenue (USD)",
            alpha=0.6,
            color="#00D9F5",
            edgecolor=None,
            s=50,
            ax=ax,
        )
        sns.regplot(data=data, x="Views", y="Estimated Revenue (USD)", scatter=False, ax=ax, color="#FF2E4D")
        ax.set_title("Views vs. Revenue (with Trendline)", pad=12, fontweight="bold")
        ax.set_xlabel("Views")
        ax.set_ylabel("Estimated Revenue (USD)")
        st.pyplot(fig)

    with col_d:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        num_cols = data.select_dtypes(include=[np.number])
        top_corr_features = num_cols.corr()["Estimated Revenue (USD)"].abs().sort_values(ascending=False).head(8).index
        sns.heatmap(data[top_corr_features].corr(), annot=True, fmt=".2f", cmap="magma", ax=ax, cbar=False)
        ax.set_title("Top Correlated Features with Revenue", pad=12, fontweight="bold")
        st.pyplot(fig)


# ─── TAB 2: Seasonal Trends ───
with tab2:
    st.markdown('<div class="section-title">Temporal & Engagement Patterns</div>', unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        if "Month" in data.columns:
            st.markdown("#### 📅 Total Revenue by Month")
            monthly_rev = data.groupby("Month")["Estimated Revenue (USD)"].sum()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            monthly_rev.plot(kind="bar", color="#7928CA", ax=ax, alpha=0.85)
            ax.set_title("Total Revenue by Month of Release", pad=12, fontweight="bold")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue (USD)")
            plt.xticks(rotation=0)
            st.pyplot(fig)
        else:
            st.info("Month column not present in dataset.")

    with col_t2:
        if "Day of Week" in data.columns:
            st.markdown("#### 📆 Revenue Generation by Day of Week")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_rev = data.groupby("Day of Week")["Estimated Revenue (USD)"].sum().reindex(day_order)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            day_rev.plot(kind="bar", color="#FF007A", ax=ax, alpha=0.85)
            ax.set_title("Revenue by Day of Week", pad=12, fontweight="bold")
            ax.set_xlabel("Day of Week")
            ax.set_ylabel("Revenue (USD)")
            plt.xticks(rotation=30)
            st.pyplot(fig)

    col_t3, col_t4 = st.columns(2)

    with col_t3:
        if "Month" in data.columns:
            st.markdown("#### 📦 Revenue Distribution Boxplot by Month")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(x="Month", y="Estimated Revenue (USD)", data=data, palette="viridis", ax=ax)
            ax.set_title("Revenue Distribution Spread by Month", pad=12, fontweight="bold")
            st.pyplot(fig)

    with col_t4:
        if "Likes" in data.columns:
            st.markdown("#### ❤️ Revenue vs. Video Likes")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.regplot(x="Likes", y="Estimated Revenue (USD)", data=data, color="#00F5A0", ax=ax, scatter_kws={"alpha":0.5})
            ax.set_title("Regression: Likes vs. Estimated Revenue", pad=12, fontweight="bold")
            st.pyplot(fig)


# ─── TAB 3: Feature Engineering ───
with tab3:
    st.markdown('<div class="section-title">Automated Feature Engineering</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        The application automatically engineers domain-specific features to boost ML predictive power:
        - **Revenue per View**: `Estimated Revenue / Views`
        - **Engagement Rate (%)**: `(Likes + Shares + Comments) / Views * 100`
        """
    )

    fe_col1, fe_col2 = st.columns(2)
    with fe_col1:
        st.markdown("#### 📊 Summary Statistics for Engineered Features")
        st.dataframe(data[["Revenue per View", "Engagement Rate"]].describe().T.style.format("{:.5f}"), use_container_width=True)

    with fe_col2:
        st.markdown("#### 🎯 Model Input Feature List")
        st.write("Features selected for Random Forest Training:")
        st.json(avail_features)


# ─── TAB 4: ML Model Engine ───
with tab4:
    st.markdown('<div class="section-title">Random Forest Regressor Performance</div>', unsafe_allow_html=True)
    
    if model_trained:
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            render_glass_card("🎯", "R² Score (Variance Explained)", f"{r2:.4f}", f"Target: > 0.85")
        with p_col2:
            render_glass_card("📉", "Root Mean Sq Error (RMSE)", f"${rmse:.4f}", "Lower is better")
        with p_col3:
            render_glass_card("📐", "Mean Absolute Error (MAE)", f"${mae:.4f}", "Avg Prediction Error")
        with p_col4:
            render_glass_card("🌳", "Trees Fitted", f"{n_estimators}", "RandomForest Hyperparam")

        st.markdown("---")
        
        m_fig1, m_fig2 = st.columns(2)

        with m_fig1:
            st.markdown("#### 📉 Prediction vs. Actual Revenue (Test Set)")
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.scatter(y_test, y_pred, alpha=0.6, color="#00F5A0", label="Test Predictions", s=45)
            min_v = min(y_test.min(), y_pred.min())
            max_v = max(y_test.max(), y_pred.max())
            ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Ideal (y = x)")
            ax.set_title("Actual vs. Predicted Revenue (USD)", pad=12, fontweight="bold")
            ax.set_xlabel("Actual Revenue (USD)")
            ax.set_ylabel("Predicted Revenue (USD)")
            ax.legend()
            st.pyplot(fig)

        with m_fig2:
            st.markdown("#### 📊 Relative Feature Importances")
            fig, ax = plt.subplots(figsize=(8, 4.8))
            feat_imp.plot(kind="barh", color="#FF2E4D", ax=ax, alpha=0.85)
            ax.set_title("Random Forest Feature Importance Weights", pad=12, fontweight="bold")
            ax.set_xlabel("Importance Weight")
            st.pyplot(fig)

    else:
        st.warning("⚠️ Insufficient columns available to train the Random Forest model.")


# ─── TAB 5: Live Revenue Simulator ───
with tab5:
    st.markdown('<div class="section-title">Interactive Revenue Prediction Simulator</div>', unsafe_allow_html=True)
    st.markdown("Adjust hypothetical video metrics below to predict anticipated revenue using the trained ML model.")

    if model_trained and rf_model is not None:
        sim_col1, sim_col2 = st.columns([1, 1])

        with sim_col1:
            st.markdown("#### 🎛️ Input Video Metrics")
            
            sim_views = st.number_input("Projected Views", min_value=100, max_value=5000000, value=25000, step=1000)
            sim_subs = st.number_input("Subscribers", min_value=10, max_value=1000000, value=500, step=50)
            sim_likes = st.number_input("Expected Likes", min_value=0, max_value=500000, value=1200, step=50)
            sim_shares = st.number_input("Expected Shares", min_value=0, max_value=50000, value=150, step=10)
            sim_comments = st.number_input("Expected New Comments", min_value=0, max_value=50000, value=80, step=5)

            # Compute engagement rate for simulation
            sim_eng_rate = ((sim_likes + sim_shares + sim_comments) / sim_views) * 100 if sim_views > 0 else 0

            st.info(f"💡 Calculated Simulated Engagement Rate: **{sim_eng_rate:.2f}%**")

        with sim_col2:
            st.markdown("#### 🔮 ML Forecast Result")
            
            sim_input_df = pd.DataFrame(
                [[sim_views, sim_subs, sim_likes, sim_shares, sim_comments, sim_eng_rate]],
                columns=avail_features,
            )

            predicted_rev = rf_model.predict(sim_input_df)[0]
            pred_rpm = (predicted_rev / sim_views) * 1000 if sim_views > 0 else 0

            st.markdown(
                f"""
                <div style="background: rgba(0, 245, 160, 0.08); border: 2px solid #00F5A0; border-radius: 20px; padding: 2rem; text-align: center; margin-top: 1rem;">
                    <div style="font-size: 0.9rem; text-transform: uppercase; color: #8B949E; letter-spacing: 1px;">Predicted Revenue</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 800; color: #00F5A0; margin: 0.5rem 0;">
                        ${predicted_rev:,.2f}
                    </div>
                    <div style="font-size: 0.95rem; color: #E6EDF3;">
                        Estimated RPM (Revenue per 1,000 Views): <strong>${pred_rpm:.2f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.markdown("<br/>", unsafe_allow_html=True)
            st.write("**Scenario Key Highlights:**")
            st.write(f"- Views Target: `{sim_views:,}`")
            st.write(f"- Likes-to-View Ratio: `{((sim_likes/sim_views)*100):.2f}%`")
            st.write(f"- Engagement Contribution: `{sim_eng_rate:.2f}%`")

    else:
        st.warning("Model simulator requires a trained model.")


# ─── TAB 6: Raw Data & Export ───
with tab6:
    st.markdown('<div class="section-title">Dataset Explorer & Predictions Download</div>', unsafe_allow_html=True)
    
    st.markdown("#### 📋 Processed Dataset")
    st.dataframe(data, use_container_width=True, height=350)

    if model_trained and y_test is not None and y_pred is not None:
        st.markdown("#### 📥 Test Set Predictions Export")
        
        pred_export_df = pd.DataFrame(
            {
                "Actual_Revenue_USD": y_test.values,
                "Predicted_Revenue_USD": y_pred,
                "Absolute_Error_USD": np.abs(y_test.values - y_pred),
            },
            index=y_test.index,
        )

        st.dataframe(pred_export_df.style.format("${:,.2f}"), use_container_width=True, height=250)

        csv_bytes = pred_export_df.to_csv().encode("utf-8")
        st.download_button(
            label="⬇️ Download Test Predictions CSV",
            data=csv_bytes,
            file_name="youtube_revenue_model_predictions.csv",
            mime="text/csv",
        )
