import os 
import pandas as pd

LOG_DIR = "C:\ALLMLPROJ\meta-overfit-predictor\logs"
OUTPUT_PATH = "C:\ALLMLPROJ\meta-overfit-predictor\meta_dataset\meta_dataset.csv"

records = []

for file_name in os.listdir(LOG_DIR):
    if not file_name.endswith(".csv"):
        continue

    run_name = file_name.replace(".csv", "")
    path = os.path.join(LOG_DIR, file_name)

    df = pd.read_csv(path)

    # ---- basic sanity check ----
    if len(df) < 5:
        continue

    # ---- final values ----
    final_train_acc = df["train_accuracy"].iloc[-1]
    final_val_acc = df["val_accuracy"].iloc[-1]
    final_gap = final_train_acc - final_val_acc

    # ---- peak validation ----
    peak_val_acc = df["val_accuracy"].max()
    peak_epoch = df["val_accuracy"].idxmax()

    # ---- after peak behavior ----
    val_after_peak = df["val_accuracy"].iloc[peak_epoch:]
    train_after_peak = df["train_accuracy"].iloc[peak_epoch:]

    val_drops = val_after_peak.iloc[-1] < peak_val_acc
    train_increases = train_after_peak.iloc[-1] > train_after_peak.iloc[0]

    # ---- labeling rules ----
    overfit = 0

    if final_gap > 0.025:
        overfit = 1
    elif final_train_acc >= 0.99 and final_val_acc < 0.97:
        overfit = 1
    elif val_drops and train_increases:
        overfit = 1

    records.append({
        "run_name": run_name,
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
        "final_gap": final_gap,
        "peak_val_accuracy": peak_val_acc,
        "overfit_label": overfit
    })

# ---- save meta dataset ----
meta_df = pd.DataFrame(records)
meta_df.to_csv(OUTPUT_PATH, index=False)

print("Saved meta dataset to:", OUTPUT_PATH)
print(meta_df)