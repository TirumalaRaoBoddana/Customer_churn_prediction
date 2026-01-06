# 📉 Customer Churn Prediction using Machine Learning

## 🔍 Overview
Customer churn prediction is a crucial problem for subscription-based and service-oriented businesses. This project focuses on building a **machine learning pipeline** to predict whether a customer is likely to churn based on historical customer data.

The project demonstrates an **end-to-end ML workflow**, including:
- Data preprocessing using `ColumnTransformer`
- Feature encoding and scaling
- Model training using scikit-learn
- Pipeline serialization using `joblib`
- Deployment using **Streamlit**

---

## 🎯 Objective
The main objectives of this project are:
- To analyze customer behavior patterns
- To predict customer churn accurately
- To build a reusable and production-ready ML pipeline
- To deploy the model as a web application

---
## 📊 Dataset
The dataset used in this project is the **E-commerce Customer Churn Analysis and Prediction Dataset**, which contains detailed customer information related to purchasing behavior, engagement, and churn patterns.

🔗 **Dataset Link (Kaggle):**  
https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

### Dataset Highlights:
- Customer demographics
- Purchase history and transaction details
- Engagement metrics and usage behavior
- Account and subscription-related attributes
- Target variable: **Churn** (Yes / No)

---
## 🧠 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Separated numerical and categorical features
- Applied:
  - `StandardScaler` for numerical features
  - `OneHotEncoder` for categorical features
- Combined preprocessing using `ColumnTransformer`

### 2️⃣ Model Training
- Trained a supervised classification model
- Wrapped preprocessing and model into a single **Pipeline**
- Ensured reproducibility and deployment safety

### 3️⃣ Model Serialization
- Saved the trained pipeline using `joblib`
- Maintained sklearn version compatibility to avoid deserialization errors

---

## 🛠️ Tech Stack
- **Programming Language**: Python 3.10
- **Libraries**:
  - scikit-learn
  - pandas
  - numpy
  - joblib
- **Deployment**: Streamlit
- **Version Control**: Git & GitHub

---

## 🚀 Deployment
The trained model is deployed using **Streamlit**, allowing users to input customer details and receive real-time churn predictions.

### ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
customer_churn_prediction/
│── app.py
│── churn_model_pipeline.pkl
│── customer-churn-prediction.ipynb
│── requirements.txt
│── README.md
```
---

## 📊 Results

  - Built a robust and reusable ML pipeline

  - Achieved reliable churn prediction performance

  - Successfully deployed the model with version-safe dependencies
## 🌟 Key Learnings

  - Importance of sklearn version compatibility when using pickle/joblib

  - Building production-ready machine learning pipelines

  - End-to-end ML deployment using Streamlit

    
---

## 📌 Future Improvements

  - Handling class imbalance using advanced techniques

  - Model explainability using SHAP or LIME
    
---

## 👨‍💻 Author

 - Boddana Tirumala Rao
  - 4th Year CSE Student | Machine Learning Enthusiast
