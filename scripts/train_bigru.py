import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_FILE = os.path.join(
    BASE_DIR, "models", "sequence_train.npz"
)

TEST_FILE = os.path.join(
    BASE_DIR, "models", "sequence_test.npz"
)

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "bigru_51class_best.pt"
)

REPORT_FILE = os.path.join(
    BASE_DIR, "results", "bigru_51class_sequence_report.txt"
)

CM_FILE = os.path.join(
    BASE_DIR, "results", "bigru_51class_confusion_matrix.png"
)

HISTORY_FILE = os.path.join(
    BASE_DIR, "results", "bigru_training_history.csv"
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_SIZE = 138
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

NUM_CLASSES = 51

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "HELLO", "YES", "NO", "THANK_YOU", "SORRY",
    "PLEASE", "HELP", "WELCOME", "WAIT", "STOP",
    "I", "YOU", "WE", "FRIEND", "FAMILY",
    "MOTHER", "FATHER", "CHILD", "DOCTOR", "NURSE",
    "HOME", "SCHOOL", "STUDENT", "WORK", "FOOD",
    "WATER", "DRINK", "EAT", "MONEY", "SLEEP",
    "COME", "GO", "GIVE", "TAKE", "WANT",
    "NEED", "KNOW", "UNDERSTAND", "ASK", "EXPLAIN",
    "HOSPITAL", "AMBULANCE", "MEDICINE", "PAIN", "SICK",
    "EMERGENCY", "DANGER", "SAFE", "ACCIDENT", "DISEASE",
    "IDLE"
]


# ============================================================
# BiGRU MODEL
# ============================================================

class BiGRUClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout
    ):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Bidirectional = 2 × hidden_size
        self.fc = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x):

        output, _ = self.bigru(x)

        # Last temporal output
        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.fc(last_output)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V4 ISL — BiGRU TEMPORAL MODEL")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

train_data = np.load(
    TRAIN_FILE,
    allow_pickle=True
)

test_data = np.load(
    TEST_FILE,
    allow_pickle=True
)

X_train = train_data["X"].astype(
    np.float32
)

y_train = train_data["y"].astype(
    np.int64
)

X_test = test_data["X"].astype(
    np.float32
)

y_test = test_data["y"].astype(
    np.int64
)


# Keep only hand + compact face = 138
X_train = X_train[:, :, :138]
X_test = X_test[:, :, :138]


print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nCalculating normalization statistics...")

train_mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

train_std = X_train.std(
    axis=(0, 1),
    keepdims=True
)

train_std[train_std < 1e-6] = 1.0

X_train = (
    X_train - train_mean
) / train_std

X_test = (
    X_test - train_mean
) / train_std

np.savez(
    "models/bigru_normalization.npz",
    mean=train_mean,
    std=train_std
)

print("Normalization parameters saved.")

# ============================================================
# DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = BiGRUClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\nMODEL")
print("Input size :", INPUT_SIZE)
print("Hidden size:", HIDDEN_SIZE)
print("Layers     :", NUM_LAYERS)
print("Bidirectional: True")
print("Dropout    :", DROPOUT)
print("Batch size :", BATCH_SIZE)
print("Epochs     :", EPOCHS)
print("LR         :", LEARNING_RATE)


# ============================================================
# TRAINING
# ============================================================

best_accuracy = 0.0
best_macro_f1 = 0.0
best_epoch = 0

history = []


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)


    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)

            pred = (
                outputs.argmax(dim=1)
                .cpu()
                .numpy()
            )

            predictions.extend(pred)
            targets.extend(
                y_batch.numpy()
            )


    test_accuracy = accuracy_score(
        targets,
        predictions
    )

    test_macro_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1
    })


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Test Acc: {test_accuracy:.4f} | "
        f"Macro F1: {test_macro_f1:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST CHECKPOINT
    # --------------------------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy
        best_macro_f1 = test_macro_f1
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "bidirectional": True,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "train_mean": train_mean,
                "train_std": train_std
            },
            MODEL_FILE
        )


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# FINAL EVALUATION
# ============================================================

final_predictions = []
final_targets = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        outputs = model(X_batch)

        pred = (
            outputs.argmax(dim=1)
            .cpu()
            .numpy()
        )

        final_predictions.extend(pred)
        final_targets.extend(
            y_batch.numpy()
        )


accuracy = accuracy_score(
    final_targets,
    final_predictions
)

macro_f1 = f1_score(
    final_targets,
    final_predictions,
    average="macro"
)

weighted_f1 = f1_score(
    final_targets,
    final_predictions,
    average="weighted"
)

report = classification_report(
    final_targets,
    final_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V4 ISL 51-CLASS BiGRU TEMPORAL MODEL\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write("DATASET\n")

    f.write(
        f"Training sequences : {len(y_train)}\n"
    )

    f.write(
        f"Testing sequences  : {len(y_test)}\n"
    )

    f.write(
        f"Frames/sequence    : {X_train.shape[1]}\n"
    )

    f.write(
        f"Features/frame     : {X_train.shape[2]}\n\n"
    )

    f.write("MODEL\n")

    f.write("BiGRU\n")

    f.write(
        f"hidden_size={HIDDEN_SIZE}\n"
    )

    f.write(
        f"num_layers={NUM_LAYERS}\n"
    )

    f.write(
        f"dropout={DROPOUT}\n"
    )

    f.write(
        f"batch_size={BATCH_SIZE}\n"
    )

    f.write(
        f"epochs={EPOCHS}\n"
    )

    f.write(
        f"learning_rate={LEARNING_RATE}\n\n"
    )

    f.write("RESULTS\n")

    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1: {weighted_f1:.4f}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    final_targets,
    final_predictions,
    labels=list(range(NUM_CLASSES))
)

plt.figure(
    figsize=(18, 16)
)

plt.imshow(cm)

plt.title(
    "BiGRU — 51-Class Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    fontsize=7
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CM_FILE,
    dpi=200
)

plt.close()


# ============================================================
# SAVE HISTORY
# ============================================================

pd.DataFrame(history).to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("RF baseline          : 89.22%")
print("GRU V1               : 87.65%")
print("GRU V2               : 85.88%")
print(
    f"BiGRU accuracy       : {accuracy * 100:.2f}%"
)
print(
    f"BiGRU Macro F1       : {macro_f1:.4f}"
)
print(
    f"BiGRU Weighted F1    : {weighted_f1:.4f}"
)
print(
    f"Best epoch           : {best_epoch}"
)

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved report:")
print(REPORT_FILE)

print("\nSaved confusion matrix:")
print(CM_FILE)

print("\nSaved training history:")
print(HISTORY_FILE)

print("=" * 70)