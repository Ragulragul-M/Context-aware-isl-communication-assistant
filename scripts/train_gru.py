import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# V4 ISL - 51 CLASS GRU TEMPORAL MODEL
# ============================================================

MODEL_DIR = Path(r".\models")
RESULTS_DIR = Path(r".\results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = MODEL_DIR / "sequence_train.npz"
TEST_FILE = MODEL_DIR / "sequence_test.npz"


# ============================================================
# 51 CLASSES
# ============================================================

SIGNS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "SORRY",
    "PLEASE",
    "HELP",
    "WELCOME",
    "WAIT",
    "STOP",
    "I",
    "YOU",
    "WE",
    "FRIEND",
    "FAMILY",
    "MOTHER",
    "FATHER",
    "CHILD",
    "DOCTOR",
    "NURSE",
    "HOME",
    "SCHOOL",
    "STUDENT",
    "WORK",
    "FOOD",
    "WATER",
    "DRINK",
    "EAT",
    "MONEY",
    "SLEEP",
    "COME",
    "GO",
    "GIVE",
    "TAKE",
    "WANT",
    "NEED",
    "KNOW",
    "UNDERSTAND",
    "ASK",
    "EXPLAIN",
    "HOSPITAL",
    "AMBULANCE",
    "MEDICINE",
    "PAIN",
    "SICK",
    "EMERGENCY",
    "DANGER",
    "SAFE",
    "ACCIDENT",
    "DISEASE",
    "IDLE"
]


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 51
INPUT_SIZE = 139
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("             V4 ISL 51-CLASS GRU MODEL")
print("=" * 70)

print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading sequence data...")

train_data = np.load(TRAIN_FILE)
test_data = np.load(TEST_FILE)

X_train = train_data["X"].astype(np.float32)
y_train = train_data["y"].astype(np.int64)

X_test = test_data["X"].astype(np.float32)
y_test = test_data["y"].astype(np.int64)

print()
print("Original shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# FEATURE NORMALIZATION
# ============================================================
#
# Calculate mean/std ONLY from training data.
# This prevents information from the test session
# leaking into training.
#
# Shape:
# X_train = (1020, 30, 139)
#
# We calculate one mean/std value for each feature.
# ============================================================

print()
print("Calculating feature normalization...")

train_2d = X_train.reshape(-1, INPUT_SIZE)

feature_mean = train_2d.mean(axis=0)
feature_std = train_2d.std(axis=0)

# Prevent division by zero
feature_std[feature_std < 1e-6] = 1.0

X_train = (
    X_train - feature_mean
) / feature_std

X_test = (
    X_test - feature_mean
) / feature_std


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
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
# GRU MODEL
# ============================================================

class ISLGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # (batch, 30, 139)

        output, hidden = self.gru(x)

        # Take the final time step
        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        logits = self.classifier(
            last_output
        )

        return logits


# ============================================================
# CREATE MODEL
# ============================================================

model = ISLGRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print()
print("=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)

print("Input size       :", INPUT_SIZE)
print("Sequence length  :", 30)
print("Hidden size      :", HIDDEN_SIZE)
print("GRU layers       :", NUM_LAYERS)
print("Dropout          :", DROPOUT)
print("Classes          :", NUM_CLASSES)
print("Batch size       :", BATCH_SIZE)
print("Epochs           :", EPOCHS)
print("Learning rate    :", LEARNING_RATE)
print("Parameters       :", total_parameters)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_test_accuracy = 0.0
best_epoch = 0

history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        logits = model(batch_X)

        loss = criterion(
            logits,
            batch_y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * batch_X.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

    train_loss = running_loss / total

    train_accuracy = (
        correct / total
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()

    test_predictions = []
    test_targets = []

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)

            logits = model(batch_X)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            test_predictions.extend(
                predictions.cpu().numpy()
            )

            test_targets.extend(
                batch_y.numpy()
            )

    test_accuracy = accuracy_score(
        test_targets,
        test_predictions
    )

    test_f1 = f1_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0
    )

    history.append(
        (
            epoch,
            train_loss,
            train_accuracy,
            test_accuracy,
            test_f1
        )
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Test Acc: {test_accuracy * 100:.2f}% | "
        f"Test F1: {test_f1:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if test_accuracy > best_test_accuracy:

        best_test_accuracy = test_accuracy
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "num_classes": NUM_CLASSES,
                "class_names": SIGNS,
                "feature_mean": feature_mean,
                "feature_std": feature_std
            },
            MODEL_DIR / "gru_51class_best.pt"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_DIR / "gru_51class_best.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# FINAL TEST
# ============================================================

final_predictions = []
final_targets = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(device)

        logits = model(batch_X)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        final_predictions.extend(
            predictions.cpu().numpy()
        )

        final_targets.extend(
            batch_y.numpy()
        )


# ============================================================
# METRICS
# ============================================================

final_accuracy = accuracy_score(
    final_targets,
    final_predictions
)

final_macro_f1 = f1_score(
    final_targets,
    final_predictions,
    average="macro",
    zero_division=0
)

final_weighted_f1 = f1_score(
    final_targets,
    final_predictions,
    average="weighted",
    zero_division=0
)

report = classification_report(
    final_targets,
    final_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=SIGNS,
    digits=4,
    zero_division=0
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("                 FINAL GRU RESULTS")
print("=" * 70)

print(
    f"Best epoch          : {best_epoch}"
)

print(
    f"Best test accuracy  : "
    f"{best_test_accuracy * 100:.2f}%"
)

print(
    f"Final test accuracy : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Macro F1            : "
    f"{final_macro_f1:.4f}"
)

print(
    f"Weighted F1         : "
    f"{final_weighted_f1:.4f}"
)

print()
print("CLASSIFICATION REPORT")
print("=" * 70)
print(report)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = (
    RESULTS_DIR /
    "gru_51class_sequence_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V4 ISL 51-CLASS GRU TEMPORAL MODEL\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write("DATASET\n")
    f.write("Training sequences : 1020\n")
    f.write("Testing sequences  : 510\n")
    f.write("Frames/sequence    : 30\n")
    f.write("Features/frame     : 139\n\n")

    f.write("MODEL\n")
    f.write("GRU\n")
    f.write(f"hidden_size={HIDDEN_SIZE}\n")
    f.write(f"num_layers={NUM_LAYERS}\n")
    f.write(f"dropout={DROPOUT}\n")
    f.write(f"batch_size={BATCH_SIZE}\n")
    f.write(f"epochs={EPOCHS}\n")
    f.write(f"learning_rate={LEARNING_RATE}\n\n")

    f.write("RESULTS\n")
    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Accuracy: {final_accuracy:.4f}\n"
    )

    f.write(
        f"Macro F1: {final_macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1: {final_weighted_f1:.4f}\n\n"
    )

    f.write("CLASSIFICATION REPORT\n")
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    final_targets,
    final_predictions,
    labels=np.arange(NUM_CLASSES)
)

plt.figure(figsize=(18, 16))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "51-Class GRU - Sequence-Level Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    np.arange(NUM_CLASSES),
    SIGNS,
    rotation=90,
    fontsize=7
)

plt.yticks(
    np.arange(NUM_CLASSES),
    SIGNS,
    fontsize=7
)

plt.colorbar()

plt.tight_layout()

cm_file = (
    RESULTS_DIR /
    "gru_51class_confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = (
    RESULTS_DIR /
    "gru_training_history.csv"
)

with open(
    history_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "epoch,train_loss,train_accuracy,"
        "test_accuracy,test_macro_f1\n"
    )

    for row in history:

        f.write(
            f"{row[0]},"
            f"{row[1]:.6f},"
            f"{row[2]:.6f},"
            f"{row[3]:.6f},"
            f"{row[4]:.6f}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("                    FINAL SUMMARY")
print("=" * 70)

print(
    f"Random Forest baseline : 89.22%"
)

print(
    f"GRU sequence accuracy  : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"GRU Macro F1           : "
    f"{final_macro_f1:.4f}"
)

print(
    f"GRU Weighted F1        : "
    f"{final_weighted_f1:.4f}"
)

print()
print("Saved model:")
print(
    MODEL_DIR /
    "gru_51class_best.pt"
)

print()
print("Saved report:")
print(report_file)

print()
print("Saved confusion matrix:")
print(cm_file)

print()
print("Saved training history:")
print(history_file)

print("=" * 70)