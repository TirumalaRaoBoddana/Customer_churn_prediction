import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

pipeline = joblib.load("churn_model_pipeline.pkl")

st.title("📊 Customer Churn Prediction App")
st.text("This application predicts whether a customer is likely to leave a service (churn) based on their information.By entering customer details such as service usage and account features, the app analyzes the data using a trained machine learning model and provides an instant prediction. The model estimates the risk of churn and classifies customers as either “Likely to Churn” or “Not Likely to Churn.")

st.divider()


with st.form("churn_form"):

    col1, col2 = st.columns(2)

   
    with col1:
        Tenure = st.number_input("Tenure (months)", min_value=0.0, step=1.0)
        Device = st.selectbox("Device", ["Mobile Phone", "Computer", "Tablet"])
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        WarehouseToHome = st.number_input("Warehouse to Home Distance", min_value=0.0)
        PaymentMode = st.selectbox(
            "Payment Mode",
            ["Debit Card", "Credit Card", "UPI", "COD", "E wallet"]
        )
        Gender = st.selectbox("Gender", ["Male", "Female"])
        HourSpendOnApp = st.number_input("Hours Spent on App", min_value=0.0)
        NumberOfDeviceRegistered = st.number_input(
            "Number of Devices Registered", min_value=1, step=1
        )
        PreferedOrderCat = st.selectbox(
            "Preferred Order Category",
            ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
        )

    with col2:
        SatisfactionScore = st.slider("Satisfaction Score", 1, 5)
        MaritalStatus = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )
        NumberOfAddress = st.number_input(
            "Number of Addresses", min_value=1, step=1
        )
        Complain = st.selectbox("Any Complaint?", [0, 1])
        OrderAmountHikeFromlastYear = st.number_input(
            "Order Amount Hike from Last Year (%)", min_value=0.0
        )
        CouponUsed = st.number_input("Coupons Used", min_value=0.0)
        OrderCount = st.number_input("Order Count", min_value=0.0)
        DaySinceLastOrder = st.number_input(
            "Days Since Last Order", min_value=0.0
        )
        CashbackAmount = st.number_input("Cashback Amount", min_value=0.0)

    submit = st.form_submit_button("🔍 Predict Churn")

if submit:

    input_data = pd.DataFrame({
        "Tenure": [Tenure],
        "Device": [Device],
        "CityTier": [CityTier],
        "WarehouseToHome": [WarehouseToHome],
        "PaymentMode": [PaymentMode],
        "Gender": [Gender],
        "HourSpendOnApp": [HourSpendOnApp],
        "NumberOfDeviceRegistered": [NumberOfDeviceRegistered],
        "PreferedOrderCat": [PreferedOrderCat],
        "SatisfactionScore": [SatisfactionScore],
        "MaritalStatus": [MaritalStatus],
        "NumberOfAddress": [NumberOfAddress],
        "Complain": [Complain],
        "OrderAmountHikeFromlastYear": [OrderAmountHikeFromlastYear],
        "CouponUsed": [CouponUsed],
        "OrderCount": [OrderCount],
        "DaySinceLastOrder": [DaySinceLastOrder],
        "CashbackAmount": [CashbackAmount]
    })
   
    churn_proba = pipeline.predict_proba(input_data)[0][1]
    churn_pred = int(churn_proba >= 0.5)
    print(churn_proba)
    st.divider()

    if churn_pred == 1:
        st.error(f"⚠️ Customer is **Likely to Churn**\n\nChurn Probability: {churn_proba}")
    else:
        st.success(f"✅ Customer is **Not Likely to Churn**\n\nChurn Probability: {churn_proba}")
