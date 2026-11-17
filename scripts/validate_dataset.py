import os
import csv
from pathlib import Path

# ============================================================
# V4 ISL DATASET VALIDATION
# ============================================================

DATASET_DIR = Path(r".\dataset")

EXPECTED_SESSIONS = 3
EXPECTED_SEQUENCES = 10
EXPECTED_FRAMES = 30
EXPECTED_COLUMNS = 141
EXPECTED_HAND = 126
EXPECTED_FACE = 12

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

# ============================================================
# COUNTERS
# ============================================================

total_sequences = 0
valid_sequences = 0
invalid_sequences = 0
total_frames = 0

problems = []

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("             V4 ISL DATASET VALIDATION")
print("=" * 70)

print(f"Dataset location : {DATASET_DIR.resolve()}")
print(f"Expected classes : {len(SIGNS)}")
print(f"Sessions/class   : {EXPECTED_SESSIONS}")
print(f"Sequences/session: {EXPECTED_SEQUENCES}")
print(f"Frames/sequence  : {EXPECTED_FRAMES}")
print("=" * 70)

# ============================================================
# CHECK DATASET DIRECTORY
# ============================================================

if not DATASET_DIR.exists():
    print("\nERROR: Dataset directory does not exist!")
    raise SystemExit

# ============================================================
# CHECK EACH SIGN
# ============================================================

for sign in SIGNS:

    sign_dir = DATASET_DIR / sign

    print(f"\n[{sign}]")

    if not sign_dir.exists():
        print("  ❌ SIGN FOLDER MISSING")
        problems.append(f"{sign}: sign folder missing")
        continue

    sign_sequence_count = 0

    # --------------------------------------------------------
    # Check sessions
    # --------------------------------------------------------

    for session_number in range(1, EXPECTED_SESSIONS + 1):

        session_name = f"session_{session_number:02d}"
        session_dir = sign_dir / session_name

        if not session_dir.exists():
            print(f"  ❌ {session_name}: MISSING")
            problems.append(f"{sign}/{session_name}: session missing")
            continue

        session_sequences = 0

        # ----------------------------------------------------
        # Check sequences
        # ----------------------------------------------------

        for sequence_number in range(1, EXPECTED_SEQUENCES + 1):

            sequence_name = f"seq_{sequence_number:03d}"
            sequence_dir = session_dir / sequence_name

            if not sequence_dir.exists():
                print(f"  ❌ {session_name}/{sequence_name}: MISSING")
                problems.append(
                    f"{sign}/{session_name}/{sequence_name}: sequence missing"
                )
                continue

            total_sequences += 1
            sign_sequence_count += 1
            session_sequences += 1

            sequence_valid = True

            # ------------------------------------------------
            # Check frames
            # ------------------------------------------------

            frames_dir = sequence_dir / "frames"

            if not frames_dir.exists():
                print(
                    f"  ❌ {session_name}/{sequence_name}: "
                    f"frames folder missing"
                )
                problems.append(
                    f"{sign}/{session_name}/{sequence_name}: frames missing"
                )
                sequence_valid = False
                frame_count = 0

            else:
                frame_files = [
                    f for f in frames_dir.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]

                frame_count = len(frame_files)

                if frame_count != EXPECTED_FRAMES:
                    print(
                        f"  ❌ {session_name}/{sequence_name}: "
                        f"{frame_count} frames (expected {EXPECTED_FRAMES})"
                    )

                    problems.append(
                        f"{sign}/{session_name}/{sequence_name}: "
                        f"{frame_count} frames"
                    )

                    sequence_valid = False

                total_frames += frame_count

            # ------------------------------------------------
            # Check features.csv
            # ------------------------------------------------

            features_file = sequence_dir / "features.csv"

            if not features_file.exists():

                print(
                    f"  ❌ {session_name}/{sequence_name}: "
                    f"features.csv missing"
                )

                problems.append(
                    f"{sign}/{session_name}/{sequence_name}: "
                    f"features.csv missing"
                )

                sequence_valid = False

            else:

                try:

                    with open(
                        features_file,
                        "r",
                        newline="",
                        encoding="utf-8"
                    ) as f:

                        reader = csv.reader(f)

                        rows = list(reader)

                    if len(rows) < 2:

                        print(
                            f"  ❌ {session_name}/{sequence_name}: "
                            f"features.csv empty"
                        )

                        problems.append(
                            f"{sign}/{session_name}/{sequence_name}: "
                            f"features.csv empty"
                        )

                        sequence_valid = False

                    else:

                        header = rows[0]
                        data_rows = rows[1:]

                        column_count = len(header)
                        row_count = len(data_rows)

                        # ----------------------------
                        # Row count
                        # ----------------------------

                        if row_count != EXPECTED_FRAMES:

                            print(
                                f"  ❌ {session_name}/{sequence_name}: "
                                f"CSV rows={row_count}, "
                                f"expected={EXPECTED_FRAMES}"
                            )

                            problems.append(
                                f"{sign}/{session_name}/{sequence_name}: "
                                f"CSV rows={row_count}"
                            )

                            sequence_valid = False

                        # ----------------------------
                        # Column count
                        # ----------------------------

                        if column_count != EXPECTED_COLUMNS:

                            print(
                                f"  ❌ {session_name}/{sequence_name}: "
                                f"columns={column_count}, "
                                f"expected={EXPECTED_COLUMNS}"
                            )

                            problems.append(
                                f"{sign}/{session_name}/{sequence_name}: "
                                f"columns={column_count}"
                            )

                            sequence_valid = False

                        # ----------------------------
                        # Hand features
                        # ----------------------------

                        hand_columns = [
                            c for c in header
                            if c.startswith("hand_")
                        ]

                        if len(hand_columns) != EXPECTED_HAND:

                            print(
                                f"  ❌ {session_name}/{sequence_name}: "
                                f"hand={len(hand_columns)}, "
                                f"expected={EXPECTED_HAND}"
                            )

                            problems.append(
                                f"{sign}/{session_name}/{sequence_name}: "
                                f"hand features={len(hand_columns)}"
                            )

                            sequence_valid = False

                        # ----------------------------
                        # Face features
                        # ----------------------------

                        face_columns = [
                            c for c in header
                            if c.startswith("face_")
                        ]

                        if len(face_columns) != EXPECTED_FACE:

                            print(
                                f"  ❌ {session_name}/{sequence_name}: "
                                f"face={len(face_columns)}, "
                                f"expected={EXPECTED_FACE}"
                            )

                            problems.append(
                                f"{sign}/{session_name}/{sequence_name}: "
                                f"face features={len(face_columns)}"
                            )

                            sequence_valid = False

                        # ----------------------------
                        # Motion
                        # ----------------------------

                        if "motion" not in header:

                            print(
                                f"  ❌ {session_name}/{sequence_name}: "
                                f"motion column missing"
                            )

                            problems.append(
                                f"{sign}/{session_name}/{sequence_name}: "
                                f"motion missing"
                            )

                            sequence_valid = False

                        # ----------------------------
                        # Frame + timestamp
                        # ----------------------------

                        if "frame" not in header:

                            print(
                                f"  ❌ {session_name}/{sequence_name}: "
                                f"frame column missing"
                            )

                            sequence_valid = False

                        if "timestamp" not in header:

                            print(
                                f"  ❌ {session_name}/{sequence_name}: "
                                f"timestamp column missing"
                            )

                            sequence_valid = False

                except Exception as e:

                    print(
                        f"  ❌ {session_name}/{sequence_name}: "
                        f"CSV read error: {e}"
                    )

                    problems.append(
                        f"{sign}/{session_name}/{sequence_name}: "
                        f"CSV read error"
                    )

                    sequence_valid = False

            # ------------------------------------------------
            # Check metadata
            # ------------------------------------------------

            metadata_file = sequence_dir / "metadata.json"

            if not metadata_file.exists():

                print(
                    f"  ❌ {session_name}/{sequence_name}: "
                    f"metadata.json missing"
                )

                problems.append(
                    f"{sign}/{session_name}/{sequence_name}: "
                    f"metadata.json missing"
                )

                sequence_valid = False

            # ------------------------------------------------
            # Final sequence status
            # ------------------------------------------------

            if sequence_valid:

                valid_sequences += 1

        # ----------------------------------------------------
        # Session summary
        # ----------------------------------------------------

        if session_sequences == EXPECTED_SEQUENCES:

            print(
                f"  ✅ {session_name}: "
                f"{session_sequences}/{EXPECTED_SEQUENCES} sequences"
            )

        else:

            print(
                f"  ⚠️ {session_name}: "
                f"{session_sequences}/{EXPECTED_SEQUENCES} sequences"
            )

    # --------------------------------------------------------
    # Sign summary
    # --------------------------------------------------------

    expected_sign_sequences = (
        EXPECTED_SESSIONS * EXPECTED_SEQUENCES
    )

    if sign_sequence_count == expected_sign_sequences:

        print(
            f"  ✅ TOTAL: "
            f"{sign_sequence_count}/{expected_sign_sequences} sequences"
        )

    else:

        print(
            f"  ⚠️ TOTAL: "
            f"{sign_sequence_count}/{expected_sign_sequences} sequences"
        )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("                    FINAL SUMMARY")
print("=" * 70)

expected_total_sequences = (
    len(SIGNS)
    * EXPECTED_SESSIONS
    * EXPECTED_SEQUENCES
)

expected_total_frames = (
    expected_total_sequences
    * EXPECTED_FRAMES
)

print(f"Classes expected       : {len(SIGNS)}")
print(f"Classes checked        : {len(SIGNS)}")
print()
print(f"Sequences expected     : {expected_total_sequences}")
print(f"Sequences found        : {total_sequences}")
print(f"Valid sequences        : {valid_sequences}")
print(f"Invalid sequences      : {total_sequences - valid_sequences}")
print()
print(f"Frames expected        : {expected_total_frames}")
print(f"Frames found           : {total_frames}")
print()
print(f"Problems found         : {len(problems)}")

# ============================================================
# PROBLEM LIST
# ============================================================

if problems:

    print("\n")
    print("=" * 70)
    print("                     PROBLEMS")
    print("=" * 70)

    for i, problem in enumerate(problems, 1):
        print(f"{i}. {problem}")

else:

    print("\n")
    print("=" * 70)
    print("              🎉 DATASET IS FULLY VALID!")
    print("=" * 70)

    print("All classes, sessions, sequences, frames,")
    print("features and metadata are present and valid.")

print("=" * 70)