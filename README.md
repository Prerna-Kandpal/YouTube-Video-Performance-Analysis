<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" alt="YouTube Logo" height="80"/>

# 📊 YouTube Video Performance Analysis

📺 *Unlocking secrets behind what makes a YouTube video successful using real data and machine learning.*

## 🔍 Overview

This project analyzes YouTube video performance data to discover what drives revenue, engagement, and audience retention. Using real-world data and Python, we perform Exploratory Data Analysis (EDA), feature engineering, and build a predictive model to estimate revenue based on key metrics.

---

## 📂 Dataset Info

🔗 **[Download Dataset from Google Drive](https://drive.google.com/file/d/10IdRG52VvMnRB6C5-a3_YqMtzOyxQnNR/view?usp=sharing)**

- **Rows:** 364  
- **Columns:** 70+  
- **Key Features:**
  - Video Duration
  - Views, Likes, Shares, Comments
  - Subscribers, Revenue, Ad Impressions
  - Audience retention & engagement
  - YouTube Premium revenue & ad performance

---

## 📈 Goals

- 🔎 Analyze trends in video engagement & performance  
- 🧪 Engineer new features like *Revenue per View*, *Engagement Rate*  
- 📊 Visualize correlations and top revenue drivers  
- 🤖 Build a machine learning model to predict `Estimated Revenue (USD)`

---

## 💻 Tech Stack

| Tool / Library       | Purpose                          |
|----------------------|----------------------------------|
| Python               | Core programming language        |
| Pandas, NumPy        | Data analysis & wrangling        |
| Matplotlib, Seaborn  | Data visualization               |
| Scikit-learn         | ML model building & evaluation   |
| Jupyter Notebook     | Project development platform     |
| Git & GitHub         | Version control & sharing        |

---

## 🧠 Key Insights

- 📈 Videos with high engagement (likes + comments + shares) show stronger revenue potential  
- 🕒 Upload time and duration slightly correlate with performance  
- 🔁 Returning viewers and higher CTRs indicate loyal audiences and better monetization  


<!--
## 📊 Visualizations

> 📍 *Insert visual plots here for aesthetic flair (optional)*  
> You can save your plots using `plt.savefig("images/revenue_distribution.png")` and then display them like this:

```
![Revenue Distribution](images/revenue_vs_views.png)
```
-->

---

## 🤖 Model Summary

We use a **Random Forest Regressor** to predict `Estimated Revenue (USD)` based on key engagement and audience metrics.

### 📏 Model Performance

| Metric               | Score  |
|----------------------|--------|
| Mean Squared Error   | ~0.45  |
| R² Score             | ~0.89  |

### 🧪 Sample Training Code

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 🧾 How to Run This Project

1. **Clone the Repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd youtube-video-performance-analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Run the notebook**: `youtube_video_performance_analysis.ipynb`

---
## 🚀 Model Enhancements & Advanced Evaluation

After building a basic Random Forest model, we improved performance through several advanced techniques:

### 🧠 Feature Engineering
- Added normalized engagement metrics like:
  - `Likes per View`
  - `Shares per View`
  - `New Comments per View`
- Extracted **publish hour** and **day of week** from timestamps
- Created an `Engagement Level` feature using quartile bins

### 🎯 Hyperparameter Tuning (Grid Search)
- Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`
- Used `GridSearchCV` with 5-fold cross-validation to find optimal values

### ⚔️ Model Comparison
We trained and compared multiple models:
- **Random Forest Regressor (Tuned)**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

#### 📊 Comparison Results (Sample)

| Model                    | R² Score | MSE    |
|--------------------------|----------|--------|
| Random Forest (Tuned)    | ~0.91    | ~0.41  |
| Gradient Boosting        | ~0.89    | ~0.44  |
| XGBoost                  | ~0.90    | ~0.42  |

### 🔁 Cross-Validation
Used 5-fold cross-validation to ensure the model generalizes well and isn't overfitting:
```python
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5, scoring='r2')
```

### 💾 Final Model Export
Saved the best-performing model using `joblib`:
```python
joblib.dump(model, "best_youtube_revenue_model.pkl")
```

---

> 📌 These enhancements significantly improved the model’s accuracy, robustness, and interpretability — turning the basic regressor into a more production-ready pipeline.

---

<div align="center">

Made with 💻 + 📊 + ☕ by **Prerna**  

</div>
