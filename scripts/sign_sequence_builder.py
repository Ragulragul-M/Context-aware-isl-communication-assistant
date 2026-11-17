from collections import deque


class SignSequenceBuilder:

    def __init__(
        self,
        confidence_threshold=0.60,
        stable_frames=5,
        idle_frames=8,
        max_sequence_length=20
    ):
        self.confidence_threshold = confidence_threshold
        self.stable_frames = stable_frames
        self.idle_frames = idle_frames
        self.max_sequence_length = max_sequence_length

        self.prediction_buffer = deque(maxlen=stable_frames)

        self.sequence = []

        self.current_sign = None
        self.idle_count = 0

    def update(self, predicted_sign, confidence):
        """
        Add one prediction from the recognition model.

        Returns:
            new_sign: a newly accepted sign, or None
        """

        # --------------------------------------------------
        # 1. Ignore low-confidence predictions
        # --------------------------------------------------
        if confidence < self.confidence_threshold:
            return None

        # --------------------------------------------------
        # 2. Handle IDLE
        # --------------------------------------------------
        if predicted_sign == "IDLE":

            self.idle_count += 1
            self.prediction_buffer.clear()

            if self.idle_count >= self.idle_frames:
                self.current_sign = None

            return None

        # --------------------------------------------------
        # 3. Non-IDLE prediction
        # --------------------------------------------------
        self.idle_count = 0

        self.prediction_buffer.append(predicted_sign)

        # --------------------------------------------------
        # 4. Wait until prediction becomes stable
        # --------------------------------------------------
        if len(self.prediction_buffer) < self.stable_frames:
            return None

        # Count occurrences
        counts = {}

        for sign in self.prediction_buffer:
            counts[sign] = counts.get(sign, 0) + 1

        stable_sign = max(counts, key=counts.get)

        # Require majority
        if counts[stable_sign] < self.stable_frames // 2 + 1:
            return None

        # --------------------------------------------------
        # 5. Prevent repeated same sign
        # --------------------------------------------------
        if stable_sign == self.current_sign:
            return None

        # --------------------------------------------------
        # 6. Accept new sign
        # --------------------------------------------------
        self.current_sign = stable_sign

        self.sequence.append(stable_sign)

        # Limit sequence length
        if len(self.sequence) > self.max_sequence_length:
            self.sequence.pop(0)

        # Clear buffer after accepting sign
        self.prediction_buffer.clear()

        return stable_sign

    def get_sequence(self):
        """Return the current recognized sign sequence."""
        return self.sequence.copy()

    def get_text(self):
        """Return sequence as text."""
        return " ".join(self.sequence)

    def reset(self):
        """Start a new sentence/sequence."""
        self.prediction_buffer.clear()
        self.sequence.clear()

        self.current_sign = None
        self.idle_count = 0


# ----------------------------------------------------------
# TEST
# ----------------------------------------------------------

if __name__ == "__main__":

    builder = SignSequenceBuilder()

    test_predictions = [

        ("YOU", 0.95),
        ("YOU", 0.94),
        ("YOU", 0.96),
        ("YOU", 0.93),
        ("YOU", 0.97),

        ("IDLE", 0.99),
        ("IDLE", 0.98),
        ("IDLE", 0.99),
        ("IDLE", 0.97),
        ("IDLE", 0.99),
        ("IDLE", 0.98),
        ("IDLE", 0.99),
        ("IDLE", 0.98),

        ("NEED", 0.91),
        ("NEED", 0.94),
        ("NEED", 0.93),
        ("NEED", 0.92),
        ("NEED", 0.95),

        ("IDLE", 0.99),
        ("IDLE", 0.99),
        ("IDLE", 0.98),
        ("IDLE", 0.99),
        ("IDLE", 0.98),
        ("IDLE", 0.99),
        ("IDLE", 0.98),
        ("IDLE", 0.99),

        ("DOCTOR", 0.96),
        ("DOCTOR", 0.95),
        ("DOCTOR", 0.97),
        ("DOCTOR", 0.94),
        ("DOCTOR", 0.96),
    ]

    for sign, confidence in test_predictions:

        new_sign = builder.update(sign, confidence)

        if new_sign:
            print(f"New sign detected: {new_sign}")

    print("\nFinal sequence:")
    print(builder.get_text())