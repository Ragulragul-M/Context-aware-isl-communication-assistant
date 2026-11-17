import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/bigru_51class_best.pt"
TEST_PATH = "models/sequence_test.npz"
NORM_PATH = "models/bigru_normalization.npz"

INPUT_SIZE = 138
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 51


# ============================================================
# LABELS
# ============================================================

LABELS = [
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
# MODEL
# ============================================================

class BiGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=0.30,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE * 2,
            NUM_CLASSES
        )

    def forward(self, x):

        output, _ = self.bigru(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD TEST DATA
# ============================================================

data = np.load(TEST_PATH)

X_test = data["X"].astype(np.float32)
y_test = data["y"]

print("Test data:", X_test.shape)
print("Labels   :", y_test.shape)


# ============================================================
# USE ONLY 138 FEATURES
# ============================================================

X_test = X_test[:, :, :138]

print("Features used:", X_test.shape)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

norm = np.load(NORM_PATH)

mean = norm["mean"].squeeze()
std = norm["std"].squeeze()

print("Mean shape:", mean.shape)
print("Std shape :", std.shape)


# ============================================================
# NORMALIZE
# ============================================================

X_test = (
    X_test - mean
) / std


# ============================================================
# LOAD MODEL
# ============================================================

model = BiGRU()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model.to(device)
model.eval()

print("BiGRU loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    outputs = model(X_tensor)

    predictions = torch.argmax(
        outputs,
        dim=1
    ).cpu().numpy()


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("=" * 60)
print("SAVED TEST SEQUENCE RESULT")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()
print(
    classification_report(
        y_test,
        predictions,
        target_names=LABELS,
        digits=4,
        zero_division=0
    )
)