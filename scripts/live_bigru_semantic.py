import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sign_sequence_builder import SignSequenceBuilder
from semantic_understanding import understand


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

        # Last time step
        last_output = output[:, -1, :]

        return self.fc(last_output)


# ============================================================
# LOAD MODEL
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

train_mean = normalization["mean"].squeeze().astype(np.float32)
train_std = normalization["std"].squeeze().astype(np.float32)

# Safety against division by zero
train_std[train_std < 1e-6] = 1.0

if train_mean.shape != (INPUT_SIZE,) or train_std.shape != (INPUT_SIZE,):
    raise ValueError(
        f"Normalization shape error: mean={train_mean.shape}, "
        f"std={train_std.shape}, expected=({INPUT_SIZE},)"
    )

print("Training normalization loaded.")
print("Mean shape:", train_mean.shape)
print("Std shape :", train_std.shape)


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
# IMPORTANT:
# This exactly matches collect_v4.py:
# LAB -> CLAHE on L -> LAB->BGR -> Gaussian Blur
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

    enhanced_lab = cv2.merge([
        enhanced_l,
        a_channel,
        b_channel
    ])

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
# IMPORTANT:
# This exactly matches collect_v4.py.
# Do NOT sort the hands by x-position.
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

            features.extend(
                [0.0] * 63
            )

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# FACE FEATURES
# IMPORTANT:
# This exactly matches collect_v4.py.
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

    # MediaPipe Face Mesh landmark indices
    left_eye_top = 159
    left_eye_bottom = 145

    right_eye_top = 386
    right_eye_bottom = 374

    mouth_top = 13
    mouth_bottom = 14

    mouth_left = 61
    mouth_right = 291

    nose = 1

    # IMPORTANT: these are the same indices used
    # during V4 dataset collection.
    left_eyebrow = 70
    right_eyebrow = 300

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

    # IMPORTANT: training collector uses the
    # midpoint of mouth_left and mouth_right.
    mouth_mid_x = (
        lm[mouth_left].x +
        lm[mouth_right].x
    ) / 2

    mouth_mid_y = (
        lm[mouth_left].y +
        lm[mouth_right].y
    ) / 2

    mouth_mid_z = (
        lm[mouth_left].z +
        lm[mouth_right].z
    ) / 2

    nose_mouth_distance = (
        np.sqrt(
            (lm[nose].x - mouth_mid_x) ** 2 +
            (lm[nose].y - mouth_mid_y) ** 2 +
            (lm[nose].z - mouth_mid_z) ** 2
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

    # Training collector normalizes head orientation
    # by eye distance.
    face_center_x = (
        lm[left_eye_outer].x +
        lm[right_eye_outer].x
    ) / 2

    head_orientation = (
        lm[nose].x -
        face_center_x
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
# MAIN CAMERA
# Use the same camera backend as collect_v4.py.
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
    raise SystemExit


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()
print("=" * 60)
print("LIVE BiGRU SIGN RECOGNITION - FIXED")
print("=" * 60)
print("Camera       :", CAMERA_INDEX)
print("Resolution   :", f"{FRAME_WIDTH}x{FRAME_HEIGHT}")
print("Sequence     :", SEQUENCE_LENGTH, "frames")
print("Features     :", INPUT_SIZE)
print("Device       :", device)
print()
print("Preprocessing: LAB CLAHE + Gaussian Blur")
print("Hand        : Exact V4 collector order")
print("Face        : Exact V4 compact features")
print("Normalization: Training mean/std")
print("Semantic    : Rule-based Stage 1")
print()
print("Press Q to quit")
print("Press R to reset sentence")
print("=" * 60)


# ============================================================
# SEQUENCE STORAGE
# ============================================================

frame_sequence = []

sequence_builder = SignSequenceBuilder(
    confidence_threshold=0.60,
    stable_frames=5,
    idle_frames=8,
    max_sequence_length=20
)

current_prediction = "WAITING..."
current_confidence = 0.0

# Semantic-understanding state
current_sign_sequence = ""
current_intent = "UNKNOWN"
current_meaning = ""


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

    # --------------------------------------------------------
    # EXACT TRAINING PREPROCESSING
    # --------------------------------------------------------

    processed = preprocess(frame)

    # MediaPipe expects RGB.
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
    ).astype(np.float32)

    # Safety check
    if features.shape != (INPUT_SIZE,):

        print(
            "Feature error:",
            features.shape
        )

        break

    frame_sequence.append(features)

    # Keep memory bounded
    if len(frame_sequence) > SEQUENCE_LENGTH:
        frame_sequence = frame_sequence[-SEQUENCE_LENGTH:]

    # --------------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------------

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in hand_results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if face_results.multi_face_landmarks:

        for face_landmarks in face_results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

    # --------------------------------------------------------
    # 30-frame sequence ready
    # --------------------------------------------------------

    if len(frame_sequence) == SEQUENCE_LENGTH:

        sequence = np.array(
            frame_sequence,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Normalize using EXACT training statistics
        # ----------------------------------------------------

        sequence = (
            sequence - train_mean
        ) / train_std

        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # Shape: (1, 30, 138)
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            sequence
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

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

        predicted_sign = LABELS[
            predicted_index
        ]

        # ----------------------------------------------------
        # Confidence threshold
        # Prevent very low-confidence predictions
        # from being added to the sentence.
        # ----------------------------------------------------

        if confidence >= CONFIDENCE_THRESHOLD:

            current_prediction = predicted_sign
            current_confidence = confidence

            sequence_builder.update(
                predicted_sign,
                confidence
            )

            # ----------------------------------------------------
            # Semantic Understanding
            # Run only when the accepted sign sequence changes.
            # ----------------------------------------------------
            new_sequence_text = sequence_builder.get_text()

            if new_sequence_text != current_sign_sequence:

                current_sign_sequence = new_sequence_text

                if current_sign_sequence.strip():

                    semantic_result = understand(
                        current_sign_sequence
                    )

                    current_intent = semantic_result["intent"]
                    current_meaning = semantic_result["natural_sentence"]

                    print()
                    print("-" * 60)
                    print("SEMANTIC UNDERSTANDING")
                    print("-" * 60)
                    print("Signs   :", semantic_result["sign_sequence"])
                    print("Intent  :", semantic_result["intent"])
                    print("Subject :", semantic_result["subject"])
                    print("Action  :", semantic_result["action"])
                    print("Object  :", semantic_result["object"])
                    print("Meaning :", semantic_result["natural_sentence"])
                    print("-" * 60)

        else:

            current_prediction = "UNCERTAIN"
            current_confidence = confidence

        # ----------------------------------------------------
        # Keep overlap for smoother temporal recognition.
        # ----------------------------------------------------

        frame_sequence = frame_sequence[-10:]

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Display recognition + semantic understanding
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Compact display overlay
    # Keep the user's face visible.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # CLEAN TEXT-ONLY DISPLAY
    # No black rectangle: keep the camera view unobstructed.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # CLEAN TEXT-ONLY DISPLAY
    # Moved upward so all lines remain visible.
    # --------------------------------------------------------

    def draw_text_clean(
        image,
        text,
        position,
        scale=0.5,
        color=(255, 255, 255),
        thickness=1
    ):
        x, y = position

        # Black outline/shadow improves readability without
        # placing a solid rectangle over the camera image.
        cv2.putText(
            image,
            text,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    sequence_text = sequence_builder.get_text()
    if not sequence_text:
        sequence_text = "---"

    if len(sequence_text) > 36:
        sequence_text = sequence_text[-36:]

    meaning_display = current_meaning if current_meaning else "---"

    if len(meaning_display) > 40:
        meaning_display = meaning_display[:37] + "..."

    # Put all information in the lower-left, but leave a
    # comfortable margin from the bottom edge.
    draw_text_clean(
        frame,
        f"Sign: {current_prediction}",
        (15, 330),
        0.58,
        (0, 255, 0),
        2
    )

    draw_text_clean(
        frame,
        f"Conf: {current_confidence:.2f}",
        (15, 355),
        0.48,
        (255, 255, 255),
        1
    )

    draw_text_clean(
        frame,
        f"Signs: {sequence_text}",
        (15, 380),
        0.46,
        (0, 255, 255),
        1
    )

    draw_text_clean(
        frame,
        f"Intent: {current_intent}",
        (15, 405),
        0.45,
        (0, 255, 255),
        1
    )

    draw_text_clean(
        frame,
        f"Meaning: {meaning_display}",
        (15, 430),
        0.40,
        (0, 255, 255),
        1
    )

    draw_text_clean(
        frame,
        f"Frames: {len(frame_sequence)}/{SEQUENCE_LENGTH}",
        (15, 453),
        0.38,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "ISL - Live BiGRU",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):

        sequence_builder.reset()
        frame_sequence.clear()

        current_prediction = "RESET"
        current_confidence = 0.0

        current_sign_sequence = ""
        current_intent = "UNKNOWN"
        current_meaning = ""

        print("\nSequence reset.")


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hands.close()
face_mesh.close()

cv2.destroyAllWindows()

print()
print("=" * 60)
print("FINAL SIGN SEQUENCE")
print("=" * 60)
print(sequence_builder.get_text())
print("=" * 60)
