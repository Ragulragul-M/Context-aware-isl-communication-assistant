import csv
from pathlib import Path

import numpy as np


# ============================================================
# V4 ISL - SEQUENCE DATA PREPARATION
# ============================================================

BASE_DIR = Path(r".\sequence_data")
OUTPUT_DIR = Path(r".\models")

TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
# EXPECTED VALUES
# ============================================================

FRAMES_PER_SEQUENCE = 30

HAND_FEATURES = 126
FACE_FEATURES = 12
MOTION_FEATURES = 1

FEATURES_PER_FRAME = (
    HAND_FEATURES
    + FACE_FEATURES
    + MOTION_FEATURES
)

EXPECTED_FEATURES = 139


# ============================================================
# READ ONE SEQUENCE
# ============================================================

def load_sequence(csv_path):

    rows = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError("CSV has no header")

        # ----------------------------------------------------
        # Select only ML features
        # ----------------------------------------------------

        hand_columns = [
            f"hand_{i}" for i in range(HAND_FEATURES)
        ]

        face_columns = [
            f"face_{i}" for i in range(FACE_FEATURES)
        ]

        feature_columns = (
            hand_columns
            + face_columns
            + ["motion"]
        )

        # ----------------------------------------------------
        # Check columns
        # ----------------------------------------------------

        missing = [
            c for c in feature_columns
            if c not in fieldnames
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        # ----------------------------------------------------
        # Read frames
        # ----------------------------------------------------

        for row in reader:

            features = []

            for column in feature_columns:

                value = row[column]

                if value == "":
                    raise ValueError(
                        f"Empty value in {column}"
                    )

                features.append(float(value))

            rows.append(features)

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    sequence = np.asarray(
        rows,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if sequence.shape != (
        FRAMES_PER_SEQUENCE,
        EXPECTED_FEATURES
    ):

        raise ValueError(
            f"Unexpected shape {sequence.shape}; "
            f"expected "
            f"({FRAMES_PER_SEQUENCE}, "
            f"{EXPECTED_FEATURES})"
        )

    # --------------------------------------------------------
    # Check NaN / Inf
    # --------------------------------------------------------

    if not np.isfinite(sequence).all():

        raise ValueError(
            "Sequence contains NaN or Inf values"
        )

    return sequence


# ============================================================
# LOAD DATASET SPLIT
# ============================================================

def load_split(split_dir, split_name):

    sequences = []
    labels = []

    errors = []

    print()
    print("=" * 70)
    print(f"LOADING {split_name.upper()} DATA")
    print("=" * 70)

    for label_index, sign in enumerate(SIGNS):

        sign_dir = split_dir / sign

        if not sign_dir.exists():

            errors.append(
                f"{sign}: class folder missing"
            )

            continue

        sequence_dirs = sorted(
            [
                d for d in sign_dir.iterdir()
                if d.is_dir()
            ]
        )

        print(
            f"{sign:<15} : "
            f"{len(sequence_dirs):>3} sequences"
        )

        for sequence_dir in sequence_dirs:

            csv_path = sequence_dir / "features.csv"

            if not csv_path.exists():

                errors.append(
                    f"{sign}/{sequence_dir.name}: "
                    f"features.csv missing"
                )

                continue

            try:

                sequence = load_sequence(
                    csv_path
                )

                sequences.append(sequence)
                labels.append(label_index)

            except Exception as e:

                errors.append(
                    f"{sign}/{sequence_dir.name}: {e}"
                )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    X = np.asarray(
        sequences,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    print()
    print(
        f"{split_name} X shape: {X.shape}"
    )

    print(
        f"{split_name} y shape: {y.shape}"
    )

    if errors:

        print()
        print("ERRORS:")

        for error in errors:
            print(" -", error)

        raise RuntimeError(
            f"{len(errors)} errors found"
        )

    return X, y


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("          V4 ISL SEQUENCE PREPARATION")
print("=" * 70)

print()
print(f"Classes              : {len(SIGNS)}")
print(f"Frames / sequence    : {FRAMES_PER_SEQUENCE}")
print(f"Hand features        : {HAND_FEATURES}")
print(f"Face features        : {FACE_FEATURES}")
print(f"Motion features      : {MOTION_FEATURES}")
print(f"Features / frame     : {EXPECTED_FEATURES}")

# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------

X_train, y_train = load_split(
    TRAIN_DIR,
    "TRAIN"
)

# ------------------------------------------------------------
# Load testing data
# ------------------------------------------------------------

X_test, y_test = load_split(
    TEST_DIR,
    "TEST"
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("                 FINAL ARRAY CHECK")
print("=" * 70)

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_test shape  : {y_test.shape}")

expected_train_shape = (
    1020,
    FRAMES_PER_SEQUENCE,
    EXPECTED_FEATURES
)

expected_test_shape = (
    510,
    FRAMES_PER_SEQUENCE,
    EXPECTED_FEATURES
)

if X_train.shape != expected_train_shape:

    raise ValueError(
        f"Training shape incorrect: "
        f"{X_train.shape}"
    )

if X_test.shape != expected_test_shape:

    raise ValueError(
        f"Testing shape incorrect: "
        f"{X_test.shape}"
    )

if y_train.shape != (1020,):

    raise ValueError(
        f"Training labels incorrect: "
        f"{y_train.shape}"
    )

if y_test.shape != (510,):

    raise ValueError(
        f"Testing labels incorrect: "
        f"{y_test.shape}"
    )

# ------------------------------------------------------------
# Check numerical values
# ------------------------------------------------------------

if not np.isfinite(X_train).all():
    raise ValueError("X_train contains NaN or Inf")

if not np.isfinite(X_test).all():
    raise ValueError("X_test contains NaN or Inf")

# ============================================================
# SAVE
# ============================================================

train_output = OUTPUT_DIR / "sequence_train.npz"
test_output = OUTPUT_DIR / "sequence_test.npz"

np.savez_compressed(
    train_output,
    X=X_train,
    y=y_train
)

np.savez_compressed(
    test_output,
    X=X_test,
    y=y_test
)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("                 CLASS DISTRIBUTION")
print("=" * 70)

for index, sign in enumerate(SIGNS):

    train_count = int(
        np.sum(y_train == index)
    )

    test_count = int(
        np.sum(y_test == index)
    )

    print(
        f"{index:2d}  {sign:<15} "
        f"Train={train_count:3d}  "
        f"Test={test_count:3d}"
    )

# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 70)
print("             🎉 PREPARATION COMPLETE")
print("=" * 70)

print(f"Training sequences : {len(X_train)}")
print(f"Testing sequences  : {len(X_test)}")

print()
print(
    "Each sequence = "
    f"{FRAMES_PER_SEQUENCE} frames × "
    f"{EXPECTED_FEATURES} features"
)

print()
print("Saved files:")
print(f"  {train_output}")
print(f"  {test_output}")

print()
print("STATUS: SUCCESS")
print("=" * 70)