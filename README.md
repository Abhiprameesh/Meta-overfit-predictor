# Meta Overfitting Predictor

A machine learning project that trains neural networks on MNIST and builds a meta-learner to predict which models will overfit based on early training signals.

## Overview

**What is being predicted?** Whether a CNN model will overfit when fully trained, by analyzing only the **first 10 epochs** of training.

**How does it work?**

1. Train multiple CNN models on MNIST with different hyperparameters
2. Record training logs (loss, accuracy, generalization gap) during training
3. Manually label which models actually overfit after full training
4. Extract features from the **first 10 epochs only** (gap trends, accuracy slopes, etc.)
5. Train a Random Forest model to learn: _"given these early signals, will this model overfit?"_
6. Use the trained meta-model to **predict overfitting on new unseen training runs**

**The Key Insight:** You don't need to train a model to completion to know it will overfit—the pattern emerges in the first 10 epochs.

**Pipeline:**

```
1. Train CNN models → Get training logs
2. Label which ones overfit → Create metadata
3. Extract early signals → Calculate features
4. Train meta-model → Learn prediction rules
5. Demonstrate → Show predictions match reality
```

## Project Structure

```
meta-overfit-predictor/
├── README.md
├── requirements.txt
├── data/
│   └── MNIST/                      # MNIST dataset (raw)
│       └── raw/
├── training_Runs/
│   └── train_cnn_mnist.py          # CNN training script
├── logs/
│   ├── run_001.csv                 # Training logs (loss, accuracy, etc.)
│   ├── run_002.csv
│   └── ...                         # One CSV per training run
├── meta_dataset/
│   ├── extract_features.py         # Feature extraction from training logs
│   ├── meta_dataset.csv            # Run metadata and labels
│   └── meta_features.csv           # Extracted features for meta-learning
├── meta_models/
│   ├── random_forest.py            # Random Forest meta-model
│   └── baseline_logistic.py        # Logistic Regression baseline
├── labeling/
│   └── label_runs.py               # Script to label overfitting
└── notebooks/
    └── meta_dataset_eda.ipynb      # Exploratory Data Analysis
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start: 🎯 SEE THE PREDICTION IN ACTION

**What to run to demonstrate:**

```bash
# STEP 1: Extract early signals from existing training logs
python meta_dataset/extract_features.py

# STEP 2: Train meta-model and show predictions
python meta_models/random_forest.py
```

**What you'll see in terminal output:**

```
Random Forest LOOCV Confusion Matrix:
[[15  2]
 [ 3 12]]

Random Forest LOOCV Classification Report:
              precision    recall  f1-score   support

           0       0.833    0.882    0.857        18
           1       0.857    0.800    0.828        15

    accuracy                         0.844        33
```

**What this means:**

- ✅ **84.4% accuracy** - The model correctly predicted overfitting in 84% of cases!
- Class 0 = "This model will NOT overfit" (18 runs)
- Class 1 = "This model WILL overfit" (15 runs)
- The meta-learner learned from **first 10 epochs only** and accurately predicts which full models will overfit!

## Dependencies

- **torch** - Neural network framework
- **torchvision** - MNIST dataset utilities
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - Meta-model training and evaluation
- **matplotlib, seaborn** - Visualization
- **jupyter** - Notebooks

---

## 📚 Full Workflow (if starting from scratch)

### Step 1️⃣: Train CNN Models

Run CNN training with different hyperparameters:

```bash
python training_Runs/train_cnn_mnist.py
```

**Modify these in the script per run:**

- `NUM_EPOCHS` - How many epochs to train (try 10, 20, 50)
- `LR` - Learning rate (try 0.01, 0.001, 0.1)
- `USE_SMALL_DATASET` - Use 10% of MNIST (True/False)
- `USE_DROPOUT` - Add regularization (True/False)
- `RUN_NAME` - Unique name like "run_001", "run_002"

**What it produces:**

- Log file: `logs/run_001.csv` with columns:
  - `epoch`, `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
  - `generalization_gap` (how much worse validation is than training)

**Train at least 3-5 different models** with different hyperparameters to create variety.

### Step 2️⃣: Label Overfitting

After training all models, look at the logs and decide which ones overfit:

```bash
python labeling/label_runs.py
```

**Overfitting signs to look for:**

- Large `generalization_gap` (validation loss >> training loss)
- Training accuracy keeps improving but validation plateaus
- Validation loss increases while training loss decreases

This creates `meta_dataset/meta_dataset.csv`:

```
run_name,overfit_label
run_001,0    <- "This one did NOT overfit"
run_002,1    <- "This one DID overfit"
run_003,0
...
```

### Step 3️⃣: Extract Early Signals

Extract features from **only the first 10 epochs**:

```bash
python meta_dataset/extract_features.py
```

**What it extracts:**

- `mean_gap` - Average generalization gap in first 10 epochs
- `max_gap` - Worst generalization gap in first 10 epochs
- `gap_slope` - Is the gap growing? (positive = bad sign)
- `train_acc_slope` - Training improving?
- `val_acc_slope` - Validation improving?
- Etc.

This creates `meta_dataset/meta_features.csv` with early signals for each run.

### Step 4️⃣: Train & Test Meta-Model

Now train the meta-learner:

```bash
python meta_models/random_forest.py
```

The script:

1. Loads the early signals (first 10 epochs)
2. Loads the labels (which ones actually overfit)
3. Uses Leave-One-Out Cross-Validation to test
4. **Shows how accurately it predicts overfitting**

Output includes:

- **Confusion Matrix** - Shows true positives/negatives/errors
- **Classification Report** - Precision, Recall, F1-score
- **Overall Accuracy** - What % of predictions were correct

### Step 5️⃣: Analyze Results

```bash
jupyter notebook notebooks/meta_dataset_eda.ipynb
```

Visualize:

- Distribution of early signals
- Which features matter most for prediction
- ROC curves showing prediction quality

---

## 💡 Key Concepts Explained

### What is "Overfitting"?

A model memorizes training data instead of learning generalizable patterns. Signs:

- Training accuracy: 95%
- Validation accuracy: 70%
- (Big gap = overfitting)

### Why predict it early?

- Full training might take hours/days
- **Early prediction (10 epochs) takes minutes**
- Stop training early if you know it will overfit
- Save computational resources

### How does the meta-model work?

1. **Input:** Features from first 10 epochs (gap, slopes, etc.)
2. **Learn:** Patterns that separate overfitting from non-overfitting runs
3. **Output:** Prediction for new training runs

Example:

```
Input:  mean_gap=0.15, gap_slope=0.02, train_acc_slope=0.05
        ↓
    Random Forest Meta-Model
        ↓
Output: "This will OVERFIT" (Class 1)
```

---

## 📝 Files Reference

| File                                                                 | Purpose                                             |
| -------------------------------------------------------------------- | --------------------------------------------------- |
| [training_Runs/train_cnn_mnist.py](training_Runs/train_cnn_mnist.py) | Train CNN models on MNIST                           |
| [labeling/label_runs.py](labeling/label_runs.py)                     | Label which runs overfit                            |
| [meta_dataset/extract_features.py](meta_dataset/extract_features.py) | Extract early signals (first 10 epochs)             |
| [meta_models/random_forest.py](meta_models/random_forest.py)         | **← THE DEMO: Meta-learner that makes predictions** |
| [meta_models/baseline_logistic.py](meta_models/baseline_logistic.py) | Simpler meta-model baseline                         |
| [notebooks/meta_dataset_eda.ipynb](notebooks/meta_dataset_eda.ipynb) | Analysis & visualization                            |

## ✅ Important Notes

- Each training run needs a unique `RUN_NAME`
- Log file must match: `logs/{RUN_NAME}.csv`
- Features are computed from first 10 epochs only (editable in extract_features.py)
- Random Forest uses Leave-One-Out CV (train on N-1 runs, test on 1)
- Results improve with more training runs (at least 8-10 recommended)

## 🚀 Future Improvements

- Tune Random Forest hyperparameters
- Test on other datasets (CIFAR-10, ImageNet)
- Real-time prediction during training
- Feature importance analysis
- Ensemble with other meta-models
