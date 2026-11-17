import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(
    BASE_DIR, "models", "sequence_train.npz"
)

TEST_FILE = os.path.join(
    BASE_DIR, "models", "sequence_test.npz"
)

RESULT_FILE = os.path.join(
    BASE_DIR, "results", "gru_ablation_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

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
# GRU MODEL
# ============================================================

class GRUClassifier(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            NUM_CLASSES
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.fc(last_output)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V4 ISL — GRU TEMPORAL ABLATION STUDY")
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

X_train_full = train_data["X"].astype(
    np.float32
)

y_train = train_data["y"].astype(
    np.int64
)

X_test_full = test_data["X"].astype(
    np.float32
)

y_test = test_data["y"].astype(
    np.int64
)


print("\nOriginal training shape:",
      X_train_full.shape)

print("Original testing shape :",
      X_test_full.shape)


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

EXPERIMENTS = {

    "A_Hand_Only": {
        "indices": list(range(0, 126))
    },

    "B_Hand_Face": {
        "indices": list(range(0, 138))
    },

    "C_Hand_Motion": {
        "indices": list(range(0, 126)) + [138]
    },

    "D_Hand_Face_Motion": {
        "indices": list(range(0, 139))
    }
}


# ============================================================
# TRAIN ONE EXPERIMENT
# ============================================================

def train_experiment(name, indices):

    print("\n")
    print("=" * 70)
    print("EXPERIMENT:", name)
    print("Features:", len(indices))
    print("=" * 70)

    X_train = X_train_full[:, :, indices]
    X_test = X_test_full[:, :, indices]

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    std[std < 1e-6] = 1.0

    X_train = (
        X_train - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = GRUClassifier(
        input_size=len(indices)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    best_accuracy = 0.0
    best_epoch = 0


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0

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

            total_loss += loss.item()


        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

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


        accuracy = accuracy_score(
            targets,
            predictions
        )

        macro_f1 = f1_score(
            targets,
            predictions,
            average="macro"
        )


        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_epoch = epoch


        if epoch % 5 == 0 or epoch == 1:

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Acc: {accuracy:.4f} | "
                f"Macro F1: {macro_f1:.4f}"
            )


    # --------------------------------------------------------
    # FINAL BEST-EPOCH VALUE
    # --------------------------------------------------------

    # Re-evaluate one final time.
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


    final_accuracy = accuracy_score(
        targets,
        predictions
    )

    final_macro_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    final_weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted"
    )


    print("\nCompleted:", name)

    print(
        f"Accuracy : {final_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1 : {final_macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {final_weighted_f1:.4f}"
    )


    return {
        "experiment": name,
        "features": len(indices),
        "accuracy": final_accuracy,
        "macro_f1": final_macro_f1,
        "weighted_f1": final_weighted_f1,
        "best_epoch": best_epoch
    }


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

results = []

for name, config in EXPERIMENTS.items():

    result = train_experiment(
        name,
        config["indices"]
    )

    results.append(result)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ABLATION SUMMARY")
print("=" * 70)

print(
    f"{'Experiment':25s}"
    f"{'Features':>10s}"
    f"{'Accuracy':>12s}"
    f"{'Macro F1':>12s}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['experiment']:25s}"
        f"{result['features']:10d}"
        f"{result['accuracy'] * 100:11.2f}%"
        f"{result['macro_f1']:12.4f}"
    )

print("-" * 70)

print("\nRF baseline: 89.22%")

print("\nSaved:")
print(RESULT_FILE)

print("=" * 70)