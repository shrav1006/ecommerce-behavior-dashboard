import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("2019-Nov2.csv", nrows=20000000)
    return df

df = load_data()

st.title("📊 Ecommerce Behavior Dashboard")

#  EDA 
st.subheader("🔍 Preview of Dataset")
st.write(df.head(20))
st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.subheader("🧼 Missing Value Summary")
st.write(df.isnull().sum())

st.subheader("📈 Event Timeline (Line Chart - Hourly)")

df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')  # safe parsing
df = df.dropna(subset=['event_time'])  # remove rows with invalid timestamps

# Resample to hourly counts
event_counts_by_hour = df.set_index('event_time').resample('H').size()

st.line_chart(event_counts_by_hour)


st.subheader("📦 Top Categories by Event")
top_categories = df['category_code'].value_counts().head(10)
st.write(top_categories)
sns.barplot(top_categories)

st.subheader("🏷 Top Brands")
top_brands = df['brand'].value_counts().head(10)
st.write(top_brands)

st.subheader("🎯 Event Type Plot (Seaborn)")
fig, ax = plt.subplots()
sns.countplot(data=df, x='event_type', order=df['event_type'].value_counts().index, ax=ax)
st.pyplot(fig)

# Customer Segmentation 

st.subheader("👥 Customer Segmentation")
df['category_code'].fillna('unknown', inplace=True)
df['brand'].fillna('unknown', inplace=True)

sampled_users = df['user_id'].drop_duplicates().sample(n=3000, random_state=42)
df_sample = df[df['user_id'].isin(sampled_users)]

user_group = df_sample.groupby('user_id').agg({
    'event_type': lambda x: x.value_counts().to_dict(),
    'price': 'mean',
    'user_session': 'nunique'
}).reset_index()

user_group[['views', 'carts', 'purchases']] = user_group['event_type'].apply(
    lambda x: [x.get('view', 0), x.get('cart', 0), x.get('purchase', 0)]
).apply(pd.Series)

# Feature engineering
user_group['view_to_cart'] = user_group['carts'] / (user_group['views'] + 1)
user_group['cart_to_purchase'] = user_group['purchases'] / (user_group['carts'] + 1)
user_group['total_events'] = user_group['views'] + user_group['carts'] + user_group['purchases']
user_group['log_price'] = np.log1p(user_group['price'])

features = user_group[['views', 'carts', 'purchases', 'price', 'user_session', 'view_to_cart','cart_to_purchase','total_events', 'log_price']]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
user_group['cluster'] = kmeans.fit_predict(scaled)

st.write("📌 Cluster-wise Summary (Averages):")
st.write(user_group.groupby('cluster')[features.columns].mean())

st.subheader("📈 Cluster-wise Purchases vs Views")
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
sns.scatterplot(data=user_group, x='views', y='purchases', hue='cluster', palette='Set2', ax=ax)
st.pyplot(fig)

#  Predictive Model
st.subheader("🤖 Predictive Modeling: Will a User Purchase?")
user_group['target'] = (user_group['purchases'] > 0).astype(int)

X = user_group[['views', 'carts', 'price', 'user_session', 'view_to_cart','cart_to_purchase', 'total_events', 'log_price']]
y = user_group['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

st.write("📉 Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.write("📋 Classification Report")
st.text(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_probs)
st.metric(label="ROC AUC Score", value=f"{roc:.2f}")

train_roc = roc_auc_score(y_res, model.predict_proba(X_res)[:, 1])
test_roc = roc_auc_score(y_test, y_probs)

st.subheader("📊 ROC AUC: Train vs Test (Check for Overfitting)")
st.write(f"✅ Train ROC AUC Score: **{train_roc:.2f}**")
st.write(f"🧪 Test ROC AUC Score: **{test_roc:.2f}**")

#Overfitting Check
if train_roc - test_roc > 0.1:
    st.warning("⚠️ Model might be overfitting. Try regularization, tuning, or simpler features.")
else:
    st.success("🎉 No major overfitting detected. Model generalizes well.")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(eval_metric='logloss', random_state=42)

scores = cross_val_score(model, X, y, scoring='roc_auc', cv=skf)
st.write("🔁 Cross-Validated ROC AUC Scores:", scores)
st.write("📊 Average CV Score:", scores.mean())

# Fit the model first
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_res, y_res)  # Make sure this is done before getting feature importance

# Then extract feature importances
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

st.subheader("🔥 Feature Importance")
st.bar_chart(feat_df.set_index("Feature"))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()

models = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': model}
results = {}

for name, clf in models.items():
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    results[name] = auc

st.subheader("📊 Model Comparison (ROC AUC)")
st.write(results)

import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

st.subheader("🔍 SHAP Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(fig)

st.subheader("""
             📌 Business Insight Summary""")

st.markdown("""
This project helps eCommerce businesses:
- 👤 Understand user behavior via **segmentation**.
- 🧠 Predict which users are likely to **purchase** using machine learning.
- 🎯 Identify key influencing features like `cart_to_purchase`, helping optimize marketing.
- 📈 Gain insight into brand and category performance via EDA.
- 🔁 Improve targeting and **personalization strategies** based on data.
""")
