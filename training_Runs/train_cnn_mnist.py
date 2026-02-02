import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import os


# -----------------------
# Experiment Config
# -----------------------
BATCH_SIZE = 64
NUM_EPOCHS = 50          # change this per run
LR = 0.05                # change this per run
USE_SMALL_DATASET = False # change this per run
USE_DROPOUT = True      # change this per run

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "../logs"
RUN_NAME = "run_006"     # increment this per run


os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------
# Dataset
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

from torch.utils.data import Subset
import random

if USE_SMALL_DATASET:
    indices = random.sample(range(len(train_dataset)), 3000)
    train_dataset = Subset(train_dataset, indices)

val_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN(use_dropout=USE_DROPOUT).to(DEVICE)

# -----------------------
# Training setup
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# -----------------------
# Training utilities
# -----------------------
def run_epoch(loader, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# -----------------------
# Training loop + logging
# -----------------------
logs = []

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    gap = train_acc - val_acc

    logs.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "generalization_gap": gap
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Gap: {gap:.4f}"
    )

# -----------------------
# Save logs
# -----------------------
df = pd.DataFrame(logs)
df.to_csv(f"{LOG_DIR}/{RUN_NAME}.csv", index=False)
