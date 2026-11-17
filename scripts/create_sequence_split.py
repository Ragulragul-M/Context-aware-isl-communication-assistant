from pathlib import Path
import shutil
import random

# ============================================================
# V4 ISL - SEQUENCE LEVEL DATASET SPLIT
# ============================================================

DATASET_DIR = Path(r".\dataset")
OUTPUT_DIR = Path(r".\sequence_data")

TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"

TRAIN_SESSIONS = ["session_01", "session_02"]
TEST_SESSIONS = ["session_03"]

# All 51 classes
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

print("=" * 70)
print("       V4 ISL SEQUENCE-LEVEL DATASET SPLIT")
print("=" * 70)

# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------

for sign in SIGNS:
    (TRAIN_DIR / sign).mkdir(parents=True, exist_ok=True)
    (TEST_DIR / sign).mkdir(parents=True, exist_ok=True)

train_count = 0
test_count = 0

# ------------------------------------------------------------
# Copy sequences
# ------------------------------------------------------------

for sign in SIGNS:

    sign_source = DATASET_DIR / sign

    if not sign_source.exists():
        print(f"ERROR: Missing sign: {sign}")
        continue

    # -------------------------
    # Training sessions
    # -------------------------

    for session in TRAIN_SESSIONS:

        session_source = sign_source / session

        if not session_source.exists():
            print(f"ERROR: Missing {sign}/{session}")
            continue

        sequences = sorted(
            [
                d for d in session_source.iterdir()
                if d.is_dir() and d.name.startswith("seq_")
            ]
        )

        for sequence in sequences:

            destination = TRAIN_DIR / sign / f"{session}_{sequence.name}"

            if destination.exists():
                shutil.rmtree(destination)

            shutil.copytree(sequence, destination)

            train_count += 1

    # -------------------------
    # Test session
    # -------------------------

    for session in TEST_SESSIONS:

        session_source = sign_source / session

        if not session_source.exists():
            print(f"ERROR: Missing {sign}/{session}")
            continue

        sequences = sorted(
            [
                d for d in session_source.iterdir()
                if d.is_dir() and d.name.startswith("seq_")
            ]
        )

        for sequence in sequences:

            destination = TEST_DIR / sign / f"{session}_{sequence.name}"

            if destination.exists():
                shutil.rmtree(destination)

            shutil.copytree(sequence, destination)

            test_count += 1

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

expected_train = 51 * 20
expected_test = 51 * 10

print()
print("=" * 70)
print("                    SPLIT SUMMARY")
print("=" * 70)

print(f"Classes              : {len(SIGNS)}")
print()
print(f"Training sessions    : {', '.join(TRAIN_SESSIONS)}")
print(f"Training sequences   : {train_count}")
print(f"Expected training    : {expected_train}")
print()
print(f"Testing sessions     : {', '.join(TEST_SESSIONS)}")
print(f"Testing sequences    : {test_count}")
print(f"Expected testing     : {expected_test}")
print()

if train_count == expected_train and test_count == expected_test:
    print("STATUS: SUCCESS")
    print("All sequences were split correctly.")
else:
    print("STATUS: ERROR")
    print("The sequence counts do not match the expected values.")

print("=" * 70)