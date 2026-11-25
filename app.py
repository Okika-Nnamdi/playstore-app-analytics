"""Google Play Store App Analytics - Streamlit app
Run with:  streamlit run app.py
Needs (same folder): installs_model.pkl, installs_features.pkl, price_model.pkl,
                     price_features.pkl, metrics.json  (all created by the notebook)
Optional (adds data charts): googleplaystore.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Play Store App Analytics", page_icon="📱", layout="wide")
BASE = Path(__file__).parent
CAT_PREFIX = "Category_Cleaned_"
CONTENT_RATINGS = ["Adults only 18+", "Everyone", "Everyone 10+", "Mature 17+", "Teen", "Unrated"]
BUCKETS = [0, 1, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000,
           1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000]


@st.cache_resource
def load_models():
    return (joblib.load(BASE / "installs_model.pkl"), list(joblib.load(BASE / "installs_features.pkl")),
            joblib.load(BASE / "price_model.pkl"), list(joblib.load(BASE / "price_features.pkl")))


@st.cache_data
def load_metrics():
    p = BASE / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_data():
    p = BASE / "googleplaystore.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["Installs"].str.contains(r"^[\d,]+\+?$", na=False)].copy()   # drops the corrupted row
    d["Installs"] = d["Installs"].str.replace("[+,]", "", regex=True).astype(int)
    d["Price"] = d["Price"].str.replace("$", "", regex=False).astype(float)
    d["Reviews"] = pd.to_numeric(d["Reviews"], errors="coerce")
    d = d.drop_duplicates(subset="App").copy()
    d["Installs_log"] = np.log1p(d["Installs"])
    return d


def installs_row(cols, category, rating_label, reviews, size, rating, price):
    row = dict.fromkeys(cols, 0)
    row.update({"Reviews_log": np.log1p(reviews), "Size_MB": size, "Rating": rating,
                "Price": price, "Type_Paid": int(price > 0)})
    for key in (f"{CAT_PREFIX}{category}", f"Content Rating_{rating_label}"):
        if key in row:            # base content rating has no column (all zeros)
            row[key] = 1
    return pd.DataFrame([row], columns=cols)


def predict_installs(model, cols, *args):
    pred = float(model.predict(installs_row(cols, *args))[0])
    return max(float(np.expm1(pred)), 0.0)


def nearest_bucket(n):
    return min(BUCKETS, key=lambda b: abs(np.log1p(b) - np.log1p(n)))


try:
    inst_model, inst_cols, price_model, price_cols = load_models()
except FileNotFoundError:
    st.error("Model files not found. Run the notebook first so it creates the `.pkl` files, "
             "then put them in the same folder as `app.py`.")
    st.stop()

metrics = load_metrics()
categories = sorted(c[len(CAT_PREFIX):] for c in inst_cols if c.startswith(CAT_PREFIX))
nice = lambda c: c.replace("_", " ").title()

st.title("📱 Play Store App Analytics")
st.caption("Predict how many installs an app might get, and what a paid app might cost.")
tab_i, tab_p, tab_d = st.tabs(["📈 Predict installs", "💲 Suggest a price", "📊 Data insights"])

# ------------------------------------------------------------ installs
with tab_i:
    left, right = st.columns([1, 1.3])
    with left:
        cat = st.selectbox("Category", categories, format_func=nice, key="ic")
        cr = st.selectbox("Content rating", CONTENT_RATINGS, index=1)
        reviews = st.number_input("Number of reviews", 0, 100_000_000, 5_000, step=500)
        size = st.slider("App size (MB)", 1.0, 100.0, 20.0)
        rating = st.slider("Average rating", 1.0, 5.0, 4.3, step=0.1, key="ir")
        price = st.number_input("Price (USD, 0 = free)", 0.0, 50.0, 0.0, step=0.5)
    with right:
        est = predict_installs(inst_model, inst_cols, cat, cr, reviews, size, rating, price)
        st.metric("Estimated installs", f"{est:,.0f}")
        st.write(f"Closest Play Store tier: **{nearest_bucket(est):,}+ installs**")
        sweep = np.unique(np.logspace(0, 6.5, 40).astype(int))
        curve = pd.DataFrame({"Reviews": sweep, "Estimated installs": [
            predict_installs(inst_model, inst_cols, cat, cr, r, size, rating, price) for r in sweep]})
        fig = px.line(curve, x="Reviews", y="Estimated installs", log_x=True, log_y=True,
                      title="Estimated installs as reviews grow (other inputs fixed)")
        fig.add_scatter(x=[max(reviews, 1)], y=[max(est, 1)], mode="markers",
                        marker=dict(size=12, color="red"), name="Your app")
        st.plotly_chart(fig, width="stretch")
        st.info("Reviews are the strongest signal in this model, so the estimate is most "
                "useful for apps that already have some reviews.")
    if metrics:
        st.caption(f"Model R² on held-out apps: {metrics['installs_r2']:.3f} (predicting log installs)")

# ------------------------------------------------------------ price
with tab_p:
    left, right = st.columns([1, 1.3])
    with left:
        pcat = st.selectbox("Category", categories, format_func=nice, key="pc")
        previews = st.number_input("Number of reviews", 0, 10_000_000, 500, step=50, key="pr")
        psize = st.slider("App size (MB)", 1.0, 100.0, 20.0, key="ps")
        prating = st.slider("Average rating", 1.0, 5.0, 4.3, step=0.1, key="prt")
    with right:
        row = dict.fromkeys(price_cols, 0)
        row.update({"Reviews_log": np.log1p(previews), "Rating": prating, "Size_MB": psize})
        if f"{CAT_PREFIX}{pcat}" in row:
            row[f"{CAT_PREFIX}{pcat}"] = 1
        suggested = float(np.clip(np.expm1(price_model.predict(pd.DataFrame([row], columns=price_cols))[0]), 0.99, 50))
        st.metric("Suggested price", f"${suggested:,.2f}")
        data = load_data()
        if data is not None:
            paid = data[(data["Type"] == "Paid") & (data["Price"] <= 50)]
            cat_paid = paid[paid["Category"] == pcat]["Price"]
            if len(cat_paid) >= 5:
                st.write(f"Median paid price in **{nice(pcat)}**: ${cat_paid.median():.2f} "
                         f"(from {len(cat_paid)} paid apps)")
            st.plotly_chart(px.histogram(paid, x="Price", nbins=40,
                                         title="Paid app prices in the dataset (up to $50)"),
                            width="stretch")
        st.warning("Be careful with this one: price is hard to predict from reviews, rating, size "
                   "and category alone. The model is only slightly better than guessing the "
                   "typical price, so treat it as a rough starting point.")
    if metrics:
        st.caption(f"Average error ${metrics['price_mae']:.2f} vs ${metrics['price_baseline_mae']:.2f} "
                   f"for always guessing the median price (R² {metrics['price_r2']:.2f}, "
                   f"{metrics['n_paid_apps_used']} paid apps).")

# ------------------------------------------------------------ insights
with tab_d:
    data = load_data()
    if data is None:
        st.info("Put `googleplaystore.csv` in this folder to see charts of the dataset.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Apps", f"{len(data):,}")
        c2.metric("Paid apps", f"{(data['Type'] == 'Paid').mean():.1%}")
        c3.metric("Median installs", f"{data['Installs'].median():,.0f}")
        top = (data.groupby("Category")["Installs_log"].mean().sort_values(ascending=False)
               .head(15).reset_index())
        top["Category"] = top["Category"].map(nice)
        st.plotly_chart(px.bar(top, x="Installs_log", y="Category", orientation="h",
                               title="Top 15 categories by average log(installs)"), width="stretch")
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(data, x="Installs_log", nbins=40,
                                    title="Distribution of log(installs)"), width="stretch")
        sample = data.dropna(subset=["Reviews"]).sample(min(2500, len(data)), random_state=1)
        b.plotly_chart(px.scatter(sample, x=np.log1p(sample["Reviews"]), y="Installs_log",
                                  opacity=0.4, labels={"x": "log(reviews)"},
                                  title="Reviews vs installs (log scale)"), width="stretch")
        st.plotly_chart(px.box(data, x="Content Rating", y="Installs_log",
                               title="log(installs) by content rating"), width="stretch")
