# 🏦 Intelligent Banking System

> An end-to-end Machine Learning system for **Loan Default Prediction**, **Customer Segmentation**, and **Product Recommendation** — built with Python, Scikit-learn, and Streamlit.

---

## 🚀 Live Demo

**👉 [Try the Live App](https://intelligentbankingsystem-yeitmp5cvkhrkfm47nkzea.streamlit.app/)**

---

## 📌 Project Overview

This project builds an intelligent banking ML pipeline that helps financial institutions:
- Predict whether a customer will **default on a loan** (Classification)
- **Segment customers** into behavioural groups using clustering (K-Means)
- **Recommend products** based on customer profile (Recommendation Engine)

---

## 🧠 Features

| Module | Algorithm | Performance |
|--------|-----------|-------------|
| Loan Default Prediction | Random Forest Classifier | 87% Accuracy |
| Customer Segmentation | K-Means Clustering | 30% targeting efficiency ↑ |
| Product Recommendation | Rule-based + ML | Personalised per segment |
| Data Insights Dashboard | EDA Visualizations | Default rate by purpose, employment, credit score |

---

## 🗂️ Project Structure

```
intelligent_banking_system/
├── Intelligent_banking_system/
│   ├── app.py                  # Streamlit web application (4 tabs)
│   ├── model_trainer.py        # ML model training & saving
│   ├── requirements.txt        # Python dependencies
│   ├── final_ds.ipynb          # EDA + model training notebook
│   └── Bank_data.csv           # Dataset
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML Libraries:** Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Tools:** Jupyter Notebook, Git, GitHub

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/cvviknesh/intelligent_banking_system.git
cd intelligent_banking_system/Intelligent_banking_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## 📊 Model Details

### Loan Default Prediction
- **Algorithm:** Random Forest Classifier
- **Accuracy:** 87%
- **Evaluation:** Confusion matrix, Classification report, ROC-AUC

### Customer Segmentation
- **Algorithm:** K-Means Clustering (k=4)
- **Segments:** Premium, Valued, Young Saver, Standard
- **Result:** 30% improvement in marketing targeting efficiency

---

## 👤 Author

**Viknesh C** — Junior Data Scientist
- 🔗 [LinkedIn](https://linkedin.com/in/viknesh01)
- 💻 [GitHub](https://github.com/cvviknesh)
- 📧 chinnaviknesh@gmail.com
