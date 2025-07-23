import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score


url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(url)
target = 'credit_risk'
X = df.drop(columns=[target])
y = df[target]

# 2. Column Separation
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. Preprocessing Pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 4. Preprocess Data
X_processed = preprocessor.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 6. Model Training
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
print("AUC Score:", roc_auc_score(y_test, y_pred_prob))
print(classification_report(y_test, y_pred))

# 8. Save Model & Preprocessor
pickle.dump(model, open("credit_model.pkl", "wb"))
pickle.dump(preprocessor, open("preprocessor.pkl", "wb"))

# 9. SHAP Explainability (Tree-based explainer avoids torch dependency)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, features=X_test[:100], show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")

# 10. Stress Testing (simulate economic downturn)
import pandas as pd

X_test_df = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

for col in X_test_df.columns:
    if 'duration' in col:
        X_test_df[col] *= 1.5
    if 'amount' in col:
        X_test_df[col] *= 1.2
    if 'age' in col:
        X_test_df[col] = (X_test_df[col] - 5).clip(lower=18)

X_stress = X_test_df.values
stress_prob = model.predict_proba(X_stress)[:, 1]

print("\n--- Stress Test Results ---")
print("AUC (Stress):", roc_auc_score(y_test, stress_prob))
print("Default Rate (Normal):", (y_pred_prob > 0.5).mean())
print("Default Rate (Stress):", (stress_prob > 0.5).mean())

# 11. Feature Importance Tracking
importances = model.feature_importances_
feat_names = preprocessor.get_feature_names_out()
importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
importance_df.sort_values(by="Importance", ascending=False).to_csv("feature_importance.csv", index=False)

top_n = 20
top_features = importance_df.sort_values(by="Importance", ascending=False).head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
plt.xlabel("Importance")
plt.title(f"Top {top_n} XGBoost Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_top20.png")
