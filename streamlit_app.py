import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load trained artifacts
model = pickle.load(open("credit_model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Load column names (from original dataset)
url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(url)
X = df.drop(columns=["credit_risk"])
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
all_cols = cat_cols + num_cols

st.set_page_config(page_title="Credit Risk Assessment", layout="centered")
st.title("Credit Risk Assessment App")
st.markdown("Predict whether a customer is likely to default or not using XGBoost and SHAP.")

# Form for input
with st.form("risk_form"):
    st.subheader(" Customer Information")
    user_input = {}
    for col in all_cols:
        if col in cat_cols:
            options = sorted(df[col].dropna().unique())
            user_input[col] = st.selectbox(f"{col}", options=options)
        else:
            user_input[col] = st.number_input(f"{col}", step=1.0, value=float(df[col].median()))
    submit = st.form_submit_button("Predict Risk")

if submit:
    # Convert to DataFrame for compatibility
    input_df = pd.DataFrame([user_input])

    # Apply preprocessing
    input_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    st.markdown("---")
    st.subheader(" Prediction Result:")
    if prediction == 1:
        st.success(f"Loan Approved — Low Default Risk (Confidence: {prob:.2f})")
    else:
        st.error(f"Loan Denied — High Default Risk (Confidence: {1 - prob:.2f})")

    # SHAP Explainability
    st.markdown("### SHAP Explanation")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_processed)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)