import os
import pandas as pd
import numpy as np

LOG_DIR = "C:\ALLMLPROJ\meta-overfit-predictor\logs"
META_PATH = "C:\ALLMLPROJ\meta-overfit-predictor\meta_dataset\meta_dataset.csv"
OUTPUT_PATH = "C:\ALLMLPROJ\meta-overfit-predictor\meta_dataset\meta_features.csv"

EARLY_EPOCHS = 10

meta_df = pd.read_csv(META_PATH)

feature_rows = []

for _, row in meta_df.iterrows():
    run_name = row["run_name"]
    label = row["overfit_label"]

    log_path = os.path.join(LOG_DIR, f"{run_name}.csv")
    df = pd.read_csv(log_path)

    if len(df) < EARLY_EPOCHS:
        continue

    early = df.iloc[:EARLY_EPOCHS]

    gap = early["generalization_gap"]
    train_acc = early["train_accuracy"]
    val_acc = early["val_accuracy"]
    train_loss = early["train_loss"]
    val_loss = early["val_loss"]

    features = {
        "run_name": run_name,
        "mean_gap": gap.mean(),
        "max_gap": gap.max(),
        "gap_slope": np.polyfit(range(EARLY_EPOCHS), gap, 1)[0],
        "train_acc_slope": np.polyfit(range(EARLY_EPOCHS), train_acc, 1)[0],
        "val_acc_slope": np.polyfit(range(EARLY_EPOCHS), val_acc, 1)[0],
        "train_loss_slope": np.polyfit(range(EARLY_EPOCHS), train_loss, 1)[0],
        "val_loss_slope": np.polyfit(range(EARLY_EPOCHS), val_loss, 1)[0],
        "early_train_acc": train_acc.iloc[-1],
        "early_val_acc": val_acc.iloc[-1],
        "overfit_label": label
    }

    feature_rows.append(features)

feature_df = pd.DataFrame(feature_rows)
feature_df.to_csv(OUTPUT_PATH, index=False)

print("Saved meta features to:", OUTPUT_PATH)
print(feature_df)
