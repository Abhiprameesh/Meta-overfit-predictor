import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report


# Load data
DATA_PATH = "C:\ALLMLPROJ\meta-overfit-predictor\meta_dataset\meta_features.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["run_name", "overfit_label"])
y = df["overfit_label"]


loo = LeaveOneOut()

y_true = []
y_pred = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    y_true.append(y_test.values[0])
    y_pred.append(pred[0])


# Evaluation
print("Random Forest LOOCV Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nRandom Forest LOOCV Classification Report:")
print(classification_report(y_true, y_pred, digits=3))


# Feature importance
model.fit(X, y)
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nRandom Forest Feature Importances:")
print(importance_df)
