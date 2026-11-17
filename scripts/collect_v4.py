import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import json
import time

# ============================================================
# V4 ISL 50-SIGN INTERACTIVE DATA COLLECTOR
# ============================================================

# ------------------------------------------------------------
# 50 SIGN CLASSES
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CAMERA_INDEX = 1

WIDTH = 640
HEIGHT = 480

FRAMES_PER_SEQUENCE = 30

# Number of sequences recorded each time the program runs
SEQUENCES_PER_RUN = 10

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

# ------------------------------------------------------------
# SELECT SIGN
# ------------------------------------------------------------

print()
print("=" * 60)
print("           V4 ISL DATA COLLECTOR")
print("=" * 60)

print()
print("AVAILABLE SIGNS")
print("-" * 60)

for i, sign in enumerate(SIGNS, start=1):
    print(f"{i:2d}. {sign}")

print("-" * 60)

while True:

    try:

        sign_number = int(
            input("Enter sign number (1-51): ")
        )

        if 1 <= sign_number <= 51:
            break

        print("Please enter a number from 1 to 50.")

    except ValueError:

        print("Please enter a valid number.")


SIGN = SIGNS[sign_number - 1]

# ------------------------------------------------------------
# SELECT SESSION
# ------------------------------------------------------------

print()
print("AVAILABLE SESSIONS")
print("-" * 60)
print("1. session_01")
print("2. session_02")
print("3. session_03")
print("-" * 60)

while True:

    try:

        session_number = int(
            input("Enter session number (1-3): ")
        )

        if 1 <= session_number <= 3:
            break

        print("Please enter 1, 2, or 3.")

    except ValueError:

        print("Please enter a valid number.")


SESSION = f"session_{session_number:02d}"

# ------------------------------------------------------------
# FINAL DATASET LOCATION
# ------------------------------------------------------------

SIGN_SESSION_DIR = os.path.join(
    DATASET_DIR,
    SIGN,
    SESSION
)

os.makedirs(
    SIGN_SESSION_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# FIND NEXT SEQUENCE NUMBER
# ------------------------------------------------------------

existing_sequences = []

for name in os.listdir(
    SIGN_SESSION_DIR
):

    if name.startswith("seq_"):

        try:

            number = int(
                name.split("_")[1]
            )

            existing_sequences.append(
                number
            )

        except ValueError:

            pass

next_sequence = (
    max(existing_sequences, default=0) + 1
)

# ------------------------------------------------------------
# MEDIAPIPE
# ------------------------------------------------------------

mp_hands = mp.solutions.hands

mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------
# DISTANCE
# ------------------------------------------------------------

def calculate_distance(a, b):

    return np.linalg.norm(
        np.array(a) - np.array(b)
    )

# ------------------------------------------------------------
# COMPACT FACE FEATURES
# ------------------------------------------------------------

def compact_face_features(face_landmarks):

    if face_landmarks is None:

        return [0.0] * 12

    points = np.array([
        [
            lm.x,
            lm.y,
            lm.z
        ]

        for lm in face_landmarks.landmark
    ])

    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    NOSE = 1

    LEFT_EYEBROW = 70
    RIGHT_EYEBROW = 300

    LEFT_EYE_CENTER = 33
    RIGHT_EYE_CENTER = 263

    eye_distance = calculate_distance(
        points[LEFT_EYE_CENTER],
        points[RIGHT_EYE_CENTER]
    )

    if eye_distance < 1e-6:

        eye_distance = 1.0

    left_eye_open = (
        calculate_distance(
            points[LEFT_EYE_TOP],
            points[LEFT_EYE_BOTTOM]
        )
        / eye_distance
    )

    right_eye_open = (
        calculate_distance(
            points[RIGHT_EYE_TOP],
            points[RIGHT_EYE_BOTTOM]
        )
        / eye_distance
    )

    average_eye_open = (
        left_eye_open +
        right_eye_open
    ) / 2.0

    mouth_open = (
        calculate_distance(
            points[MOUTH_TOP],
            points[MOUTH_BOTTOM]
        )
        / eye_distance
    )

    mouth_width = (
        calculate_distance(
            points[MOUTH_LEFT],
            points[MOUTH_RIGHT]
        )
        / eye_distance
    )

    mouth_aspect_ratio = (
        mouth_open /
        (mouth_width + 1e-6)
    )

    nose_mouth_distance = (
        calculate_distance(
            points[NOSE],
            (
                points[MOUTH_LEFT] +
                points[MOUTH_RIGHT]
            ) / 2
        )
        / eye_distance
    )

    face_width = (
        calculate_distance(
            points[LEFT_EYE_CENTER],
            points[RIGHT_EYE_CENTER]
        )
        / eye_distance
    )

    left_eyebrow_distance = (
        calculate_distance(
            points[LEFT_EYEBROW],
            points[LEFT_EYE_CENTER]
        )
        / eye_distance
    )

    right_eyebrow_distance = (
        calculate_distance(
            points[RIGHT_EYEBROW],
            points[RIGHT_EYE_CENTER]
        )
        / eye_distance
    )

    average_eyebrow_elevation = (
        left_eyebrow_distance +
        right_eyebrow_distance
    ) / 2.0

    nose_x = points[NOSE][0]

    face_center_x = (
        points[LEFT_EYE_CENTER][0] +
        points[RIGHT_EYE_CENTER][0]
    ) / 2.0

    head_orientation = (
        nose_x -
        face_center_x
    ) / eye_distance

    return [
        float(left_eye_open),
        float(right_eye_open),
        float(average_eye_open),
        float(mouth_open),
        float(mouth_width),
        float(mouth_aspect_ratio),
        float(nose_mouth_distance),
        float(face_width),
        float(left_eyebrow_distance),
        float(right_eyebrow_distance),
        float(average_eyebrow_elevation),
        float(head_orientation)
    ]

# ------------------------------------------------------------
# HAND FEATURES
# ------------------------------------------------------------

def extract_hand_features(hand_results):

    features = []

    detected_hands = []

    if hand_results.multi_hand_landmarks:

        detected_hands = list(
            hand_results.multi_hand_landmarks
        )

    for hand_index in range(2):

        if hand_index < len(
            detected_hands
        ):

            landmarks = (
                detected_hands[
                    hand_index
                ]
            )

            wrist = (
                landmarks.landmark[0]
            )

            wrist_x = wrist.x
            wrist_y = wrist.y
            wrist_z = wrist.z

            for lm in landmarks.landmark:

                features.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])

        else:

            features.extend(
                [0.0] * 63
            )

    return features

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

def preprocess(frame):

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = (
        cv2.split(lab)
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        [
            enhanced_l,
            a_channel,
            b_channel
        ]
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    return blurred

# ------------------------------------------------------------
# OPTICAL FLOW
# ------------------------------------------------------------

previous_gray = None

def calculate_motion(processed_frame):

    global previous_gray

    gray = cv2.cvtColor(
        processed_frame,
        cv2.COLOR_BGR2GRAY
    )

    if previous_gray is None:

        previous_gray = gray.copy()

        return 0.0

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    magnitude, angle = (
        cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )
    )

    average_motion = float(
        magnitude.mean()
    )

    previous_gray = gray.copy()

    return average_motion

# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

if not cap.isOpened():

    print()
    print("ERROR: Could not open camera.")

    hands.close()
    face_mesh.close()

    raise SystemExit

# ------------------------------------------------------------
# INFORMATION
# ------------------------------------------------------------

print()
print("=" * 60)
print("SELECTED CONFIGURATION")
print("=" * 60)

print(f"Sign              : {SIGN}")
print(f"Session           : {SESSION}")
print(f"Starting sequence : {next_sequence}")
print(f"Sequences         : {SEQUENCES_PER_RUN}")
print(f"Frames/sequence   : {FRAMES_PER_SEQUENCE}")
print(f"Resolution        : {WIDTH} x {HEIGHT}")

print()
print("Pipeline:")
print("Preprocessing")
print("Hand landmarks    = 126")
print("Compact face      = 12")
print("Fusion             = 138")
print("Optical flow       = motion")
print("=" * 60)

print()
print("Controls:")
print("R = Record next sequence")
print("Q = Quit")
print("=" * 60)

# ------------------------------------------------------------
# COLLECTION LOOP
# ------------------------------------------------------------

sequence_count = 0

while sequence_count < SEQUENCES_PER_RUN:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read camera."
        )

        break

    processed = preprocess(
        frame
    )

    rgb = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2RGB
    )

    hand_results = hands.process(
        rgb
    )

    face_results = face_mesh.process(
        rgb
    )

    hand_features = (
        extract_hand_features(
            hand_results
        )
    )

    if face_results.multi_face_landmarks:

        face_features = (
            compact_face_features(
                face_results.multi_face_landmarks[0]
            )
        )

    else:

        face_features = [0.0] * 12

    motion = calculate_motion(
        processed
    )

    # --------------------------------------------------------
    # DRAW HAND
    # --------------------------------------------------------

    display = processed.copy()

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in (
            hand_results.multi_hand_landmarks
        ):

            mp_drawing.draw_landmarks(
                display,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # --------------------------------------------------------
    # DRAW FACE
    # --------------------------------------------------------

    if face_results.multi_face_landmarks:

        for face_landmarks in (
            face_results.multi_face_landmarks
        ):

            mp_drawing.draw_landmarks(
                display,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                None,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=1
                )
            )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"SIGN: {SIGN}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        display,
        f"SESSION: {SESSION}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Sequence: {sequence_count + 1}/{SEQUENCES_PER_RUN}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "R = Record    Q = Quit",
        (15, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "V4 ISL Data Collector",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    if key == ord("r"):

        sequence_number = (
            next_sequence +
            sequence_count
        )

        sequence_dir = os.path.join(
            SIGN_SESSION_DIR,
            f"seq_{sequence_number:03d}"
        )

        frames_dir = os.path.join(
            sequence_dir,
            "frames"
        )

        os.makedirs(
            frames_dir,
            exist_ok=True
        )

        features_file = os.path.join(
            sequence_dir,
            "features.csv"
        )

        metadata_file = os.path.join(
            sequence_dir,
            "metadata.json"
        )

        print()
        print(
            f"Recording {SIGN} "
            f"{SESSION} "
            f"Sequence {sequence_number}"
        )

        # ----------------------------------------------------
        # COUNTDOWN
        # ----------------------------------------------------

        for countdown in [3, 2, 1]:

            ret, countdown_frame = (
                cap.read()
            )

            if not ret:
                break

            countdown_display = (
                countdown_frame.copy()
            )

            cv2.putText(
                countdown_display,
                f"GET READY: {countdown}",
                (150, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

            cv2.imshow(
                "V4 ISL Data Collector",
                countdown_display
            )

            cv2.waitKey(1000)

        # ----------------------------------------------------
        # CSV HEADER
        # ----------------------------------------------------

        header = [
            "frame",
            "timestamp"
        ]

        header += [
            f"hand_{i}"
            for i in range(126)
        ]

        header += [
            f"face_{i}"
            for i in range(12)
        ]

        header += [
            "motion"
        ]

        # ----------------------------------------------------
        # CAPTURE
        # ----------------------------------------------------

        timestamps = []

        with open(
            features_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow(
                header
            )

            sequence_start = time.time()

            captured_frames = 0

            for frame_number in range(
                1,
                FRAMES_PER_SEQUENCE + 1
            ):

                ret, frame = (
                    cap.read()
                )

                if not ret:

                    print(
                        "Frame capture failed."
                    )

                    break

                processed = preprocess(
                    frame
                )

                rgb = cv2.cvtColor(
                    processed,
                    cv2.COLOR_BGR2RGB
                )

                hand_results = (
                    hands.process(rgb)
                )

                face_results = (
                    face_mesh.process(rgb)
                )

                hand_features = (
                    extract_hand_features(
                        hand_results
                    )
                )

                if face_results.multi_face_landmarks:

                    face_features = (
                        compact_face_features(
                            face_results.multi_face_landmarks[0]
                        )
                    )

                else:

                    face_features = [
                        0.0
                    ] * 12

                motion = (
                    calculate_motion(
                        processed
                    )
                )

                timestamp = (
                    time.time() -
                    sequence_start
                )

                timestamps.append(
                    timestamp
                )

                # Save original frame
                frame_path = os.path.join(
                    frames_dir,
                    f"frame_{frame_number:03d}.jpg"
                )

                cv2.imwrite(
                    frame_path,
                    frame
                )

                # Save features
                writer.writerow(
                    [
                        frame_number,
                        timestamp
                    ]
                    +
                    hand_features
                    +
                    face_features
                    +
                    [motion]
                )

                captured_frames += 1

                # Display
                recording_display = (
                    processed.copy()
                )

                if hand_results.multi_hand_landmarks:

                    for hand_landmarks in (
                        hand_results.multi_hand_landmarks
                    ):

                        mp_drawing.draw_landmarks(
                            recording_display,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                if face_results.multi_face_landmarks:

                    for face_landmarks in (
                        face_results.multi_face_landmarks
                    ):

                        mp_drawing.draw_landmarks(
                            recording_display,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION
                        )

                cv2.putText(
                    recording_display,
                    f"RECORDING: {SIGN}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    recording_display,
                    f"{SESSION}  Sequence {sequence_number}",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    recording_display,
                    f"Frame: {frame_number}/{FRAMES_PER_SEQUENCE}",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "V4 ISL Data Collector",
                    recording_display
                )

                cv2.waitKey(1)

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = {

            "label": SIGN,

            "session": SESSION,

            "sequence_id":
                sequence_number,

            "num_frames":
                captured_frames,

            "resolution": [
                WIDTH,
                HEIGHT
            ],

            "camera_index":
                CAMERA_INDEX,

            "hand_features":
                126,

            "compact_face_features":
                12,

            "fusion_features":
                138,

            "motion_feature":
                1,

            "preprocessing": {

                "clahe_clip_limit":
                    2.0,

                "clahe_tile_grid":
                    [8, 8],

                "gaussian_kernel":
                    [3, 3]
            }
        }

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        sequence_count += 1

        print()
        print(
            f"Saved sequence "
            f"{sequence_number}"
        )

        print(
            f"Frames: "
            f"{captured_frames}"
        )

        print(
            f"Location: "
            f"{sequence_dir}"
        )

        time.sleep(1)

# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()

hands.close()

face_mesh.close()

cv2.destroyAllWindows()

print()
print("=" * 60)
print("COLLECTION FINISHED")
print("=" * 60)

print(
    f"Sign    : {SIGN}"
)

print(
    f"Session : {SESSION}"
)

print(
    f"Sequences recorded: "
    f"{sequence_count}"
)

print(
    f"Location: "
    f"{SIGN_SESSION_DIR}"
)

print("=" * 60)