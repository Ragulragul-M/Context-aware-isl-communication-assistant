import cv2
import mediapipe as mp
import numpy as np
import csv
import os


# ============================================================
# CONFIG
# ============================================================

CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FRAMES_TO_CAPTURE = 30

OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "live_features_138.csv"
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


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


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    return cv2.cvtColor(
        blurred,
        cv2.COLOR_GRAY2RGB
    )


# ============================================================
# HAND FEATURES
# ============================================================

def extract_hand_features(results):

    features = []

    if results.multi_hand_landmarks:

        detected = []

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = []

            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:

                x = lm.x - wrist.x
                y = lm.y - wrist.y
                z = lm.z - wrist.z

                landmarks.extend([
                    x, y, z
                ])

            detected.append(landmarks)

        # Current live-script ordering
        detected.sort(
            key=lambda x: x[0]
        )

        for hand in detected:
            features.extend(hand)

    while len(features) < 126:
        features.append(0.0)

    return np.array(
        features[:126],
        dtype=np.float32
    )


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


# ============================================================
# FACE FEATURES
# ============================================================

def extract_face_features(results):

    if not results.multi_face_landmarks:

        return np.zeros(
            12,
            dtype=np.float32
        )

    lm = results.multi_face_landmarks[0].landmark

    left_eye_top = 159
    left_eye_bottom = 145

    right_eye_top = 386
    right_eye_bottom = 374

    mouth_top = 13
    mouth_bottom = 14

    mouth_left = 61
    mouth_right = 291

    nose = 1

    left_eyebrow = 105
    right_eyebrow = 334

    left_eye_outer = 33
    right_eye_outer = 263

    eye_distance = distance(
        lm[left_eye_outer],
        lm[right_eye_outer]
    )

    if eye_distance < 1e-6:
        eye_distance = 1.0

    left_eye_open = (
        distance(
            lm[left_eye_top],
            lm[left_eye_bottom]
        ) / eye_distance
    )

    right_eye_open = (
        distance(
            lm[right_eye_top],
            lm[right_eye_bottom]
        ) / eye_distance
    )

    avg_eye_open = (
        left_eye_open +
        right_eye_open
    ) / 2

    mouth_open = (
        distance(
            lm[mouth_top],
            lm[mouth_bottom]
        ) / eye_distance
    )

    mouth_width = (
        distance(
            lm[mouth_left],
            lm[mouth_right]
        ) / eye_distance
    )

    mouth_aspect_ratio = (
        mouth_open /
        (mouth_width + 1e-6)
    )

    nose_mouth_distance = (
        distance(
            lm[nose],
            lm[mouth_top]
        ) / eye_distance
    )

    face_width = (
        distance(
            lm[left_eye_outer],
            lm[right_eye_outer]
        ) / eye_distance
    )

    left_eyebrow_distance = (
        distance(
            lm[left_eyebrow],
            lm[left_eye_top]
        ) / eye_distance
    )

    right_eyebrow_distance = (
        distance(
            lm[right_eyebrow],
            lm[right_eye_top]
        ) / eye_distance
    )

    avg_eyebrow_elevation = (
        left_eyebrow_distance +
        right_eyebrow_distance
    ) / 2

    head_orientation = (
        lm[nose].x -
        (
            lm[left_eye_outer].x +
            lm[right_eye_outer].x
        ) / 2
    )

    return np.array(
        [
            left_eye_open,
            right_eye_open,
            avg_eye_open,
            mouth_open,
            mouth_width,
            mouth_aspect_ratio,
            nose_mouth_distance,
            face_width,
            left_eyebrow_distance,
            right_eyebrow_distance,
            avg_eyebrow_elevation,
            head_orientation
        ],
        dtype=np.float32
    )


# ============================================================
# CAMERA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

if not cap.isOpened():

    raise RuntimeError(
        "Camera could not be opened."
    )


print()
print("=" * 60)
print("LIVE FEATURE DIAGNOSTIC")
print("=" * 60)
print()
print("Press R to start capturing 30 frames.")
print("Press Q to quit.")
print()


capturing = False
features_list = []


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    rgb = preprocess(frame)

    hand_results = hands.process(rgb)

    face_results = face_mesh.process(rgb)

    hand_features = extract_hand_features(
        hand_results
    )

    face_features = extract_face_features(
        face_results
    )

    features = np.concatenate(
        [
            hand_features,
            face_features
        ]
    )

    # --------------------------------------------------------
    # Capture
    # --------------------------------------------------------

    if capturing:

        features_list.append(
            features
        )

        cv2.putText(
            frame,
            f"CAPTURING: {len(features_list)}/30",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if len(features_list) >= FRAMES_TO_CAPTURE:

            X = np.array(
                features_list,
                dtype=np.float32
            )

            np.savetxt(
                OUTPUT_FILE,
                X,
                delimiter=","
            )

            print()
            print("Capture complete!")
            print("Shape:", X.shape)
            print("Saved:", OUTPUT_FILE)

            break

    else:

        cv2.putText(
            frame,
            "Press R to capture",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Live Feature Diagnostic",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r") and not capturing:

        capturing = True

        features_list = []

        print("Capturing 30 frames...")

    elif key == ord("q"):

        break


cap.release()

hands.close()
face_mesh.close()

cv2.destroyAllWindows()