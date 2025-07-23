# Credit Risk Assessment App

A machine learning web application that predicts whether a customer is a **credit risk** (likely to default) using **XGBoost** and **SHAP** explainability. Built with **Streamlit** and trained on the German Credit dataset.

---

## Features

- Binary classification (`0` = default, `1` = good credit)
- Fully pipelined preprocessing (numerical + categorical)
- XGBoost classifier with SHAP-based feature attribution
- Interactive Streamlit UI with real-time prediction and explanation
- Feature importance & stress testing included
