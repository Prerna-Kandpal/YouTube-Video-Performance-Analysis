# 📊 YouTube Video Performance Analytics & Revenue Predictor

A state-of-the-art, production-ready **Streamlit** dashboard with a modern dark-mode glassmorphic interface that analyzes YouTube channel data, performs in-depth EDA, and uses a **Random Forest Regressor** to predict video revenue in real-time.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![UI Design](https://img.shields.io/badge/UI-Glassmorphism-00F5A0)

---

## 🚀 Live Demo

Deploy this application on **Streamlit Cloud** with a single click:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ UI & Feature Highlights

- 🎨 **Glassmorphic Cyberpunk UI**: Built with custom CSS, Google Fonts (`Outfit` & `Inter`), translucent frosted glass cards, glowing hover borders, and rich gradient banners.
- 📁 **Smart Dataset Handler**: Automatically loads the included `youtube_channel_data.csv` or allows custom CSV file uploads via sidebar.
- 📊 **Executive Overview Cards**: Live metrics for Total Channel Revenue, Views, Average CTR, Engagement Rate, Subscribers, and Model $R^2$ Score.
- 📈 **Interactive Visualizations**:
  - Revenue Distribution Histograms
  - Top 10 Revenue Generating Videos
  - Views vs. Revenue Scatter Plot with Trendlines
  - Feature Correlation Heatmaps
  - Monthly & Daily Revenue Bar Charts and Boxplots
  - Regression plots for Revenue vs. Likes
- 🔮 **Live Video Revenue Simulator**: Interactive calculator tool where users can adjust projected Views, Subscribers, Likes, Shares, and Comments to predict estimated earnings in real-time.
- 📥 **Predictions Export**: Export test set predictions and error analysis as CSV files.

---

## 📂 Repository Structure

```
.
├── app.py                                # Main Streamlit Web Application
├── requirements.txt                      # Python Dependencies
├── youtube_channel_data.csv              # Default CSV Dataset
├── youtube_video_performance_analysis.ipynb # Original Notebook
└── README.md                             # Project Documentation
```

---

## 🛠️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate   # On Linux/macOS
.venv\Scripts\activate      # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py
```

The app will launch automatically at **http://localhost:8501**.

---

## ☁️ Streamlit Cloud Deployment

1. Push all files (`app.py`, `requirements.txt`, `youtube_channel_data.csv`, `README.md`) to your GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select your repository.
4. Set `app.py` as the main file path and click **Deploy**.

---

## 📝 License

Distributed under the [MIT License](LICENSE).
