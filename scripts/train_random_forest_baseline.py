import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt


# ============================================================
# V4 ISL - RANDOM FOREST FRAME-LEVEL BASELINE
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("       V4 ISL RANDOM FOREST BASELINE")
print("=" * 70)

print("\nLoading sequence data...")

train_data = np.load(TRAIN_FILE)
test_data = np.load(TEST_FILE)

X_train_seq = train_data["X"]
y_train_seq = train_data["y"]

X_test_seq = test_data["X"]
y_test_seq = test_data["y"]

print()
print("Original sequence shapes:")
print("X_train:", X_train_seq.shape)
print("y_train:", y_train_seq.shape)
print("X_test :", X_test_seq.shape)
print("y_test :", y_test_seq.shape)


# ============================================================
# CONVERT SEQUENCES TO FRAME-LEVEL DATA
# ============================================================

print("\nConverting sequences to frame-level training data...")

# 1020 sequences × 30 frames × 139 features
num_train_sequences = X_train_seq.shape[0]
num_test_sequences = X_test_seq.shape[0]

num_frames = X_train_seq.shape[1]
num_features = X_train_seq.shape[2]

# Flatten only for the Random Forest baseline.
# Temporal order is retained separately for majority voting.

X_train_frames = X_train_seq.reshape(
    num_train_sequences * num_frames,
    num_features
)

X_test_frames = X_test_seq.reshape(
    num_test_sequences * num_frames,
    num_features
)

# Repeat each sequence label 30 times
y_train_frames = np.repeat(
    y_train_seq,
    num_frames
)

y_test_frames = np.repeat(
    y_test_seq,
    num_frames
)

print()
print("Frame-level shapes:")
print("X_train_frames:", X_train_frames.shape)
print("y_train_frames:", y_train_frames.shape)
print("X_test_frames :", X_test_frames.shape)
print("y_test_frames :", y_test_frames.shape)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Parameters:")
print("n_estimators     :", 500)
print("min_samples_leaf :", 2)
print("max_features     :", "sqrt")
print("class_weight     :", "balanced")
print("random_state     :", 42)

print("\nTraining...")

model.fit(
    X_train_frames,
    y_train_frames
)

print("Training completed.")


# ============================================================
# SAVE MODEL
# ============================================================

model_file = MODEL_DIR / "rf_51class_frame_baseline.pkl"

joblib.dump(
    model,
    model_file
)

print("\nModel saved:")
print(model_file)


# ============================================================
# FRAME-LEVEL PREDICTION
# ============================================================

print("\nGenerating frame-level predictions...")

y_pred_frames = model.predict(
    X_test_frames
)

frame_accuracy = accuracy_score(
    y_test_frames,
    y_pred_frames
)

frame_f1_macro = f1_score(
    y_test_frames,
    y_pred_frames,
    average="macro",
    zero_division=0
)

frame_f1_weighted = f1_score(
    y_test_frames,
    y_pred_frames,
    average="weighted",
    zero_division=0
)

print()
print("=" * 70)
print("FRAME-LEVEL RESULTS")
print("=" * 70)

print(
    f"Accuracy       : {frame_accuracy * 100:.2f}%"
)

print(
    f"Macro F1       : {frame_f1_macro:.4f}"
)

print(
    f"Weighted F1    : {frame_f1_weighted:.4f}"
)


# ============================================================
# CONVERT FRAME PREDICTIONS BACK INTO SEQUENCES
# ============================================================

print("\nPerforming sequence-level majority voting...")

# Reshape:
# 510 sequences × 30 frame predictions

pred_sequence_frames = y_pred_frames.reshape(
    num_test_sequences,
    num_frames
)


# ------------------------------------------------------------
# Majority vote
# ------------------------------------------------------------

y_pred_sequence = []

for sequence_predictions in pred_sequence_frames:

    counts = np.bincount(
        sequence_predictions,
        minlength=len(SIGNS)
    )

    predicted_class = np.argmax(counts)

    y_pred_sequence.append(
        predicted_class
    )

y_pred_sequence = np.asarray(
    y_pred_sequence,
    dtype=np.int64
)


# ============================================================
# SEQUENCE-LEVEL RESULTS
# ============================================================

sequence_accuracy = accuracy_score(
    y_test_seq,
    y_pred_sequence
)

sequence_f1_macro = f1_score(
    y_test_seq,
    y_pred_sequence,
    average="macro",
    zero_division=0
)

sequence_f1_weighted = f1_score(
    y_test_seq,
    y_pred_sequence,
    average="weighted",
    zero_division=0
)

print()
print("=" * 70)
print("SEQUENCE-LEVEL RESULTS")
print("=" * 70)

print(
    f"Accuracy       : {sequence_accuracy * 100:.2f}%"
)

print(
    f"Macro F1       : {sequence_f1_macro:.4f}"
)

print(
    f"Weighted F1    : {sequence_f1_weighted:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test_seq,
    y_pred_sequence,
    labels=np.arange(len(SIGNS)),
    target_names=SIGNS,
    digits=4,
    zero_division=0
)

print()
print("=" * 70)
print("SEQUENCE-LEVEL CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = RESULTS_DIR / "rf_51class_sequence_report.txt"

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V4 ISL RANDOM FOREST 51-CLASS BASELINE\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write("DATASET\n")
    f.write(f"Training sequences : {num_train_sequences}\n")
    f.write(f"Testing sequences  : {num_test_sequences}\n")
    f.write(f"Frames/sequence    : {num_frames}\n")
    f.write(f"Features/frame     : {num_features}\n\n")

    f.write("MODEL\n")
    f.write("Random Forest\n")
    f.write("n_estimators=500\n")
    f.write("min_samples_leaf=2\n")
    f.write("max_features=sqrt\n")
    f.write("class_weight=balanced\n")
    f.write("random_state=42\n\n")

    f.write("FRAME-LEVEL RESULTS\n")
    f.write(f"Accuracy: {frame_accuracy:.4f}\n")
    f.write(f"Macro F1: {frame_f1_macro:.4f}\n")
    f.write(f"Weighted F1: {frame_f1_weighted:.4f}\n\n")

    f.write("SEQUENCE-LEVEL RESULTS\n")
    f.write(f"Accuracy: {sequence_accuracy:.4f}\n")
    f.write(f"Macro F1: {sequence_f1_macro:.4f}\n")
    f.write(f"Weighted F1: {sequence_f1_weighted:.4f}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test_seq,
    y_pred_sequence,
    labels=np.arange(len(SIGNS))
)

plt.figure(figsize=(18, 16))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "51-Class Random Forest - Sequence-Level Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    np.arange(len(SIGNS)),
    SIGNS,
    rotation=90,
    fontsize=7
)

plt.yticks(
    np.arange(len(SIGNS)),
    SIGNS,
    fontsize=7
)

plt.colorbar()

plt.tight_layout()

cm_file = RESULTS_DIR / "rf_51class_confusion_matrix.png"

plt.savefig(
    cm_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("                 FINAL SUMMARY")
print("=" * 70)

print(
    f"Frame-level accuracy     : "
    f"{frame_accuracy * 100:.2f}%"
)

print(
    f"Sequence-level accuracy  : "
    f"{sequence_accuracy * 100:.2f}%"
)

print(
    f"Sequence-level Macro F1  : "
    f"{sequence_f1_macro:.4f}"
)

print(
    f"Sequence-level Weighted F1: "
    f"{sequence_f1_weighted:.4f}"
)

print()
print("Saved model:")
print(model_file)

print()
print("Saved report:")
print(report_file)

print()
print("Saved confusion matrix:")
print(cm_file)

print("=" * 70)