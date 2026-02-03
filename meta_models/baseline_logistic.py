import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut

# -----------------------
# Load data
# -----------------------
DATA_PATH = "C:\ALLMLPROJ\meta-overfit-predictor\meta_dataset\meta_features.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["run_name", "overfit_label"])
y = df["overfit_label"]

# -----------------------
# Pipeline: scaling + model
# -----------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="liblinear",
        random_state=42
    ))
])

# -----------------------
# Train model
# -----------------------
pipeline.fit(X, y)

# -----------------------
# Predictions
# -----------------------
y_pred = pipeline.predict(X)
y_prob = pipeline.predict_proba(X)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred, digits=3))


feature_names = X.columns
coefficients = pipeline.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values(by="coefficient", ascending=False)

print("\nLogistic Regression Coefficients:")
print(coef_df)

# -----------------------
# LOOCV setup
# -----------------------
loo = LeaveOneOut()

y_true = []
y_pred = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    y_true.append(y_test.values[0])
    y_pred.append(pred[0])

# -----------------------
# Evaluation
# -----------------------
print("LOOCV Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nLOOCV Classification Report:")
print(classification_report(y_true, y_pred, digits=3))