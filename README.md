An end-to-end machine learning application that predicts whether a customer is likely to default on credit using the German Credit Dataset. The model is trained using XGBoost and made explainable with SHAP. A Streamlit UI allows users to input customer data and see risk predictions with visual explanations.

Features

Predict whether a customer is a credit risk (0 = default, 1 = safe)

Uses an XGBoost classifier trained on cleaned and preprocessed data

Automatically handles numeric and categorical features via pipelines

SHAP explanations for model transparency (waterfall plots)

