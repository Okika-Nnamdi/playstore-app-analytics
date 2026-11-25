# 📱 Google Play Store App Analytics

Predicts how many **installs** a Play Store app is likely to get, and suggests a **price** for paid apps, with an interactive Streamlit dashboard.

**🔗 Live demo:** _add your Streamlit link here after deploying_
**👤 Author:** [Okika Nnamdi](https://github.com/Okika-Nnamdi)

![App demo](images/demo.gif)

## 📌 Problem
Developers want to know what installs to expect and how to price an app. This project cleans the public Google Play Store dataset, explores what drives installs and price, and trains two regression models.

## 📊 Dataset
- **Source:** Kaggle - *Google Play Store Apps* (`googleplaystore.csv`)
- **Size:** 10,841 raw rows, **9,659 unique apps** after cleaning

## 🧹 Data cleaning
| Issue | Fix |
|---|---|
| One corrupted row with shifted columns | Removed |
| Missing ratings | Filled with the median rating |
| `Size` as text ("19M", "201k", "Varies with device") | Converted to megabytes (`Size_MB`); "Varies" filled with the median |
| `Installs` and `Price` as text ("10,000+", "$4.99") | Symbols stripped and converted to numbers |
| `Last Updated` | Split into day, month, year |
| `Current Ver`, `Android Ver` | Reduced to the major version number |
| Duplicate apps | Kept the first row per app name |
| Skewed installs and reviews | `log(1 + x)` transform |
| Category | One-hot encoded |

## 🤖 Models
| Task | Model | Result (held-out 25%) |
|---|---|---|
| Predict `log(installs)` | Linear regression | **R² = 0.94** |
| Predict price of paid apps | Linear regression on `log(price)` | **MAE $2.46** (guessing the median price gives $2.54), R² 0.11 |

**Features:** reviews (log), app size, rating, category, content rating, type and price for installs; reviews (log), rating, size and category for price.

## ⚠️ Honest limitations
- **Installs:** the high R² comes mostly from the number of reviews, which grows together with installs. The model works best for apps that already have reviews and is not a forecast for a brand-new app.
- **Price:** price is hard to predict from these features. The model is only slightly better than always guessing the typical price, so treat its output as a rough starting point. Paid apps above $50 (such as the $399.99 novelty apps) were excluded because they distorted the model. Random forest and gradient boosting were tried and did not help much.
- Data is a 2018 snapshot of the Play Store.

## 🖥️ App
- **Predict installs:** estimate and nearest Play Store tier, with a curve showing how installs grow with reviews
- **Suggest a price:** suggested price with the category's median paid price for comparison
- **Data insights:** top categories, install distribution, reviews vs installs, content rating

## 🚀 Run locally
```bash
git clone https://github.com/Okika-Nnamdi/playstore-app-analytics.git
cd playstore-app-analytics
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
1. Open `App_Installs_and_Price_Prediction_Fixed.ipynb` and **Run All**. This creates the `.pkl` files and `metrics.json`.
2. Start the app: `streamlit run app.py`

## 📁 Structure
```
├── app.py
├── App_Installs_and_Price_Prediction_Fixed.ipynb
├── googleplaystore.csv
├── installs_model.pkl / installs_features.pkl
├── price_model.pkl / price_features.pkl
├── metrics.json
└── requirements.txt
```

## 🛠️ Tech stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit

## 📬 Contact
Okika Nnamdi · okikannamdi18@gmail.com · [GitHub](https://github.com/Okika-Nnamdi)
