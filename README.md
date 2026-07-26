# 📊 YouTube Video Performance Analytics

A production-ready **Streamlit** dashboard that analyses real-world YouTube channel data, performs exploratory data analysis, and uses a **Random Forest Regressor** to predict estimated video revenue.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)

---

## 🚀 Live Demo

Deploy this app on **Streamlit Cloud** in one click:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ Features

| Feature | Description |
|---|---|
| **CSV Upload** | Upload your YouTube analytics CSV directly through the UI |
| **Dataset Preview** | View shape, types, null counts, and a scrollable preview |
| **Auto Cleaning** | Drops rows with missing values and reports the delta |
| **Rich EDA** | Revenue distribution, Top 10 videos, Views vs Revenue scatter, Correlation heatmap |
| **Feature Engineering** | Automatically creates *Revenue per View* and *Engagement Rate* |
| **ML Model** | Trains a Random Forest Regressor with configurable hyper-parameters |
| **Evaluation Metrics** | Displays R² Score, RMSE, and MAE in glass-style cards |
| **Visualisations** | Prediction vs Actual, Feature Importance, Monthly & Daily Revenue, Boxplots, Regression plots |
| **Download** | Export predictions as CSV |

---

## 📂 Project Structure

```
.
├── app.py               # Streamlit application (single-file)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**. Upload the YouTube analytics CSV via the sidebar.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select this repo → set `app.py` as the main file.
4. Click **Deploy**.

> **No additional configuration is required.** The `requirements.txt` is automatically detected.

---

## 📊 Dataset

The app expects a CSV with YouTube channel/video performance columns such as:

- `Views`, `Watch Time (hours)`, `Subscribers`
- `Likes`, `Shares`, `New Comments`
- `Estimated Revenue (USD)`
- `Impressions`, `Video Thumbnail CTR (%)`
- `Video Duration`, `Average View Duration`, `Average View Percentage (%)`
- `Month`, `Day of Week`, `ID`

A sample dataset (`youtube_channel_real_performance_analytics.csv`) is available in the repository (if included) or can be uploaded at runtime.

---

## 🤖 Model Details

| Parameter | Default |
|---|---|
| Algorithm | Random Forest Regressor |
| Number of Trees | 100 (configurable via sidebar) |
| Test Split | 20% (configurable via sidebar) |
| Random State | 42 (configurable via sidebar) |

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
