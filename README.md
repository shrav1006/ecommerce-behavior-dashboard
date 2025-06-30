# 🛒 Ecommerce Behavior Analysis Dashboard

An interactive Streamlit dashboard to analyze and predict customer behavior from a large-scale ecommerce dataset. This project includes exploratory data analysis (EDA), customer segmentation, and predictive modeling using machine learning.

---

## 📌 Why This Project?

Understanding online customer behavior is essential for any ecommerce platform. This dashboard helps answer critical business questions:
- Who are the potential buyers?
- What behaviors lead to a purchase?
- Which products and brands are most popular?
- How can we use data to boost sales?

---

## 🚀 Features

### 1. **Exploratory Data Analysis (EDA)**
- Preview of 20M+ ecommerce events
- Hourly event timeline visualization
- Top categories & brands
- Event type distribution with plots

### 2. **Customer Segmentation**
- User-level aggregation of views, carts, and purchases
- Feature engineering: `view_to_cart`, `cart_to_purchase`, etc.
- KMeans clustering with interactive visuals

### 3. **Predictive Modeling**
- Target: Will a user purchase or not?
- Features: event behavior, price, sessions, and more
- Model: XGBoost with SMOTE to handle class imbalance
- Evaluation: Confusion matrix, classification report, ROC AUC score
- Model comparison: Logistic Regression, Random Forest, XGBoost
- Explainability: SHAP feature importance

---

## 🧠 Tech Stack

- **Language**: Python
- **Dashboard**: Streamlit
- **Visualization**: Seaborn, Matplotlib
- **ML Models**: XGBoost, Logistic Regression, Random Forest
- **Preprocessing**: Scikit-learn, SMOTE
- **Explainability**: SHAP

---

## 🛠 How to Run This Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/shrav1006/ecommerce-behavior-dashboard.git
   cd ecommerce-behavior-dashboard

2. **Dataset Placeholder**
    The original dataset is too large to upload to GitHub.
    To run this project:
    1. Download the full dataset from the original source (e.g., Kaggle or provided link).
    2. Place the file as: `./data/2019-Nov2.csv`

3. **Install dependencies** : pip install -r requirements.txt

4. **Run the app** : streamlit run app.py

---
## Folder Structure 

ecommerce-behavior-dashboard/
 app.py                  # Main Streamlit dashboard code
 2019-Nov2.csv           # Ecommerce dataset (~67M rows)
 requirements.txt        # Python dependencies
 README.md               # You're reading it!


## License

This project is **not open source** and is intended for personal or educational use only.  
No permission is granted to copy, reproduce, distribute, or use any part of this code or content.

⚠️ Note: Due to size limits, the full dataset is not included. Use the provided sample or follow instructions in the `/data/` folder to add the full dataset locally.
