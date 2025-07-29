# 🎬 YouTube Video Performance Analysis

This project explores real-world YouTube video metrics to uncover insights about viewership, engagement, and monetization. It uses exploratory data analysis (EDA) and machine learning to understand the drivers of revenue and predict estimated earnings.

---

##  Dataset Overview

- **Data Range**: 2016 – 2024  
- **Source**: YouTube Analytics (exported or simulated)  
- **Key Columns**:
  - `Views`, `Subscribers`, `Likes`, `Shares`, `New Comments`
  - `Estimated Revenue (USD)`
  - `Video Duration`, `Video Publish Time`
  - `Ad Impressions`, `CPM`, `CTR`, etc.

---

## 🔍 Project Workflow

###  Step 1: Data Cleaning
- Removed missing/null values
- Converted date and time formats
- Filtered numeric columns
- Created:
  - `Revenue per View`
  - `Engagement Rate`

###  Step 2: Exploratory Data Analysis (EDA)
- Pairplots of revenue vs views, subscribers, likes
- Distribution plot of estimated revenue
- Top 10 highest revenue-generating videos
- Revenue vs Views scatter plot
- Correlation heatmap

### Step 3: Machine Learning
- **Model**: Random Forest Regressor  
- **Target**: `Estimated Revenue (USD)`  
- **Features**: `Views`, `Subscribers`, `Likes`, `Shares`, `New Comments`, `Engagement Rate`  
- **Evaluation**:
  - Mean Squared Error (MSE)
  - R² Score
- **Saved Model**: `youtube_revenue_predictor.pkl`

###  Step 4: Key Insights
- **Views** and **Engagement Rate** are the biggest revenue drivers  
- Videos with more **shares**, **likes**, and **comments** perform better  
- High CTR and watch time improve monetization

---

## 🛠️ Tools & Libraries

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Joblib  
- Jupyter Notebook

---
