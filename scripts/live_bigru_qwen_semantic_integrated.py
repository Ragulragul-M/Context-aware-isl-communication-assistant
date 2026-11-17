import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from sign_sequence_builder import SignSequenceBuilder
from vlm_semantic_understanding import understand_sign_sequence


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SEQUENCE_LENGTH = 30
INPUT_SIZE = 138

MODEL_PATH = "models/bigru_51class_best.pt"
NORMALIZATION_PATH = "models/bigru_normalization.npz"

CONFIDENCE_THRESHOLD = 0.50

# Live sequence-builder settings that were already working.
BUILDER_CONFIDENCE = 0.60
STABLE_FRAMES = 2
IDLE_FRAMES = 8
MAX_SEQUENCE_LENGTH = 20


# ============================================================
# 51 CLASS LABELS
# ============================================================

LABELS = [
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
# BIGRU MODEL
# ============================================================

class BiGRU(nn.Module):

    def __init__(
        self,
        input_size=138,
        hidden_size=128,
        num_layers=2,
        num_classes=51,
        dropout=0.30
    ):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x):
        output, _ = self.bigru(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)


# ============================================================
# LOAD BIGRU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = BiGRU(
    input_size=INPUT_SIZE,
    hidden_size=128,
    num_layers=2,
    num_classes=51,
    dropout=0.30
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("BiGRU model loaded successfully.")


# ============================================================
# LOAD TRAINING NORMALIZATION
# ============================================================

normalization = np.load(NORMALIZATION_PATH)

train_mean = normalization["mean"].squeeze()
train_std = normalization["std"].squeeze()

if train_mean.shape != (INPUT_SIZE,):
    raise ValueError(
        f"Normalization mean shape is {train_mean.shape}; "
        f"expected {(INPUT_SIZE,)}"
    )

if train_std.shape != (INPUT_SIZE,):
    raise ValueError(
        f"Normalization std shape is {train_std.shape}; "
        f"expected {(INPUT_SIZE,)}"
    )

print("BiGRU normalization loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

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


# ============================================================
# PREPROCESSING
# IMPORTANT: aligned with the V4 collector
# ============================================================

def preprocess(frame):

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

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


# ============================================================
# HAND FEATURES
# IMPORTANT: same ordering as the V4 collector
# ============================================================

def extract_hand_features(hand_results):

    features = []

    detected_hands = []

    if hand_results.multi_hand_landmarks:
        detected_hands = list(
            hand_results.multi_hand_landmarks
        )

    for hand_index in range(2):

        if hand_index < len(detected_hands):

            landmarks = detected_hands[hand_index]
            wrist = landmarks.landmark[0]

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
            features.extend([0.0] * 63)

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# FACE FEATURES
# IMPORTANT: same compact 12-feature definition as V4
# ============================================================

def distance(a, b):

    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def extract_face_features(face_results):

    if not face_results.multi_face_landmarks:
        return np.zeros(
            12,
            dtype=np.float32
        )

    lm = face_results.multi_face_landmarks[0].landmark

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

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    eye_distance = distance(
        lm[LEFT_EYE_OUTER],
        lm[RIGHT_EYE_OUTER]
    )

    if eye_distance < 1e-6:
        eye_distance = 1.0

    left_eye_open = (
        distance(
            lm[LEFT_EYE_TOP],
            lm[LEFT_EYE_BOTTOM]
        ) / eye_distance
    )

    right_eye_open = (
        distance(
            lm[RIGHT_EYE_TOP],
            lm[RIGHT_EYE_BOTTOM]
        ) / eye_distance
    )

    avg_eye_open = (
        left_eye_open +
        right_eye_open
    ) / 2.0

    mouth_open = (
        distance(
            lm[MOUTH_TOP],
            lm[MOUTH_BOTTOM]
        ) / eye_distance
    )

    mouth_width = (
        distance(
            lm[MOUTH_LEFT],
            lm[MOUTH_RIGHT]
        ) / eye_distance
    )

    mouth_aspect_ratio = (
        mouth_open /
        (mouth_width + 1e-6)
    )

    # Match the V4 collector: nose to mouth midpoint.
    mouth_mid_x = (
        lm[MOUTH_LEFT].x +
        lm[MOUTH_RIGHT].x
    ) / 2.0

    mouth_mid_y = (
        lm[MOUTH_LEFT].y +
        lm[MOUTH_RIGHT].y
    ) / 2.0

    mouth_mid_z = (
        lm[MOUTH_LEFT].z +
        lm[MOUTH_RIGHT].z
    ) / 2.0

    nose_mouth_distance = (
        np.sqrt(
            (lm[NOSE].x - mouth_mid_x) ** 2 +
            (lm[NOSE].y - mouth_mid_y) ** 2 +
            (lm[NOSE].z - mouth_mid_z) ** 2
        ) / eye_distance
    )

    # This is normalized by eye distance, therefore approximately 1.
    face_width = (
        distance(
            lm[LEFT_EYE_OUTER],
            lm[RIGHT_EYE_OUTER]
        ) / eye_distance
    )

    left_eyebrow_distance = (
        distance(
            lm[LEFT_EYEBROW],
            lm[LEFT_EYE_TOP]
        ) / eye_distance
    )

    right_eyebrow_distance = (
        distance(
            lm[RIGHT_EYEBROW],
            lm[RIGHT_EYE_TOP]
        ) / eye_distance
    )

    avg_eyebrow_elevation = (
        left_eyebrow_distance +
        right_eyebrow_distance
    ) / 2.0

    head_orientation = (
        lm[NOSE].x -
        (
            lm[LEFT_EYE_OUTER].x +
            lm[RIGHT_EYE_OUTER].x
        ) / 2.0
    ) / eye_distance

    features = [
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
    ]

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
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
    print("ERROR: Camera could not be opened.")
    hands.close()
    face_mesh.close()
    raise SystemExit


# ============================================================
# SIGN SEQUENCE BUILDER
# ============================================================

sequence_builder = SignSequenceBuilder(
    confidence_threshold=BUILDER_CONFIDENCE,
    stable_frames=STABLE_FRAMES,
    idle_frames=IDLE_FRAMES,
    max_sequence_length=MAX_SEQUENCE_LENGTH
)

frame_sequence = []

current_prediction = "WAITING..."
current_confidence = 0.0

current_sign_sequence = ""
current_intent = ""
current_meaning = ""

semantic_busy = False


def update_semantic_if_needed():

    global current_sign_sequence
    global current_intent
    global current_meaning
    global semantic_busy

    new_sequence = sequence_builder.get_text().strip()

    if not new_sequence:
        return

    if new_sequence == current_sign_sequence:
        return

    # Prevent overlapping semantic calls.
    if semantic_busy:
        return

    semantic_busy = True

    try:
        print()
        print("=" * 70)
        print("NEW SIGN SEQUENCE")
        print("=" * 70)
        print(new_sequence)

        result = understand_sign_sequence(
            new_sequence
        )

        current_sign_sequence = new_sequence
        current_intent = str(
            result.get("intent") or "UNKNOWN"
        )

        current_meaning = str(
            result.get("natural_sentence") or ""
        )

        print("Intent   :", current_intent)
        print("Meaning  :", current_meaning)
        print("=" * 70)

    except Exception as error:

        print(
            "Semantic processing error:",
            error
        )

    finally:
        semantic_busy = False


# ============================================================
# TEXT DRAWING HELPER
# ============================================================

def draw_text(
    image,
    text,
    position,
    scale=0.55,
    color=(255, 255, 255),
    thickness=2
):

    # Black outline/shadow for readability without a rectangle.
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("LIVE ISL + BiGRU + SEMANTIC UNDERSTANDING")
print("=" * 70)
print("Camera       :", CAMERA_INDEX)
print("Resolution   :", f"{FRAME_WIDTH}x{FRAME_HEIGHT}")
print("BiGRU input  :", INPUT_SIZE)
print("Sequence     :", SEQUENCE_LENGTH, "frames")
print("BiGRU device :", device)
print("Semantic     : Qwen2.5-3B-Instruct")
print()
print("Q = Quit")
print("R = Reset sign sequence")
print("=" * 70)


# ============================================================
# LIVE LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame failed.")
        break

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    processed = preprocess(frame)

    rgb = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

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

    if features.shape[0] != INPUT_SIZE:
        print(
            "Feature error:",
            features.shape
        )
        break

    frame_sequence.append(features)

    # --------------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------------

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in (
            hand_results.multi_hand_landmarks
        ):

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if face_results.multi_face_landmarks:

        for face_landmarks in (
            face_results.multi_face_landmarks
        ):

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

    # --------------------------------------------------------
    # BiGRU prediction
    # --------------------------------------------------------

    if len(frame_sequence) >= SEQUENCE_LENGTH:

        sequence = np.array(
            frame_sequence[-SEQUENCE_LENGTH:],
            dtype=np.float32
        )

        sequence = (
            sequence - train_mean
        ) / train_std

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        confidence = confidence.item()
        predicted_index = predicted_index.item()

        predicted_sign = LABELS[predicted_index]

        if confidence >= CONFIDENCE_THRESHOLD:
            current_prediction = predicted_sign
        else:
            current_prediction = "UNCERTAIN"

        current_confidence = confidence

        # ----------------------------------------------------
        # Sequence Builder
        # ----------------------------------------------------

        previous_sequence = (
            sequence_builder.get_text().strip()
        )

        sequence_builder.update(
            predicted_sign,
            confidence
        )

        new_sequence = (
            sequence_builder.get_text().strip()
        )

        # Run semantic understanding only when the accepted
        # sign sequence actually changes.
        if new_sequence != previous_sequence:
            update_semantic_if_needed()

        # Keep the established rolling-window behavior.
        frame_sequence = frame_sequence[-10:]

    # --------------------------------------------------------
    # LIVE UI
    # --------------------------------------------------------

    sequence_text = (
        sequence_builder.get_text().strip()
    )

    draw_text(
        frame,
        f"Sign: {current_prediction}",
        (20, 330),
        scale=0.70,
        color=(0, 255, 0),
        thickness=2
    )

    draw_text(
        frame,
        f"Confidence: {current_confidence:.2f}",
        (20, 355),
        scale=0.55,
        color=(255, 255, 255),
        thickness=2
    )

    draw_text(
        frame,
        f"Signs: {sequence_text[-55:]}",
        (20, 380),
        scale=0.55,
        color=(255, 255, 0),
        thickness=2
    )

    draw_text(
        frame,
        f"Intent: {current_intent}",
        (20, 405),
        scale=0.55,
        color=(0, 255, 255),
        thickness=2
    )

    meaning_display = current_meaning[-65:]

    draw_text(
        frame,
        f"Meaning: {meaning_display}",
        (20, 430),
        scale=0.55,
        color=(0, 255, 255),
        thickness=2
    )

    draw_text(
        frame,
        f"Frames: {len(frame_sequence)}/{SEQUENCE_LENGTH}",
        (20, 453),
        scale=0.50,
        color=(255, 255, 255),
        thickness=2
    )

    cv2.imshow(
        "ISL - Live BiGRU + Semantic",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):
        break

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    elif key == ord("r"):

        sequence_builder.reset()

        frame_sequence.clear()

        current_prediction = "RESET"
        current_confidence = 0.0

        current_sign_sequence = ""
        current_intent = ""
        current_meaning = ""

        print("\nSequence and semantic context reset.")


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hands.close()
face_mesh.close()

cv2.destroyAllWindows()

print()
print("=" * 70)
print("FINAL SIGN SEQUENCE")
print("=" * 70)
print(sequence_builder.get_text())
print()
print("FINAL INTENT :", current_intent)
print("FINAL MEANING:", current_meaning)
print("=" * 70)
