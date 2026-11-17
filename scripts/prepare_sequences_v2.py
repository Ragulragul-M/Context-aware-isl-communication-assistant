import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(BASE_DIR, "models", "sequence_train.npz")
TEST_FILE = os.path.join(BASE_DIR, "models", "sequence_test.npz")

OUT_TRAIN = os.path.join(BASE_DIR, "models", "sequence_train_v2.npz")
OUT_TEST = os.path.join(BASE_DIR, "models", "sequence_test_v2.npz")


def add_temporal_features(X):
    """
    Input:
        X shape = (N, T, 139)

    Existing features:
        0:126   -> hand landmarks
        126:138 -> compact face
        138     -> optical-flow motion

    New temporal features:
        hand velocity
        hand acceleration
        motion direction/change

    Output:
        richer temporal feature sequence
    """

    hand = X[:, :, :126]
    face = X[:, :, 126:138]
    flow_motion = X[:, :, 138:139]

    # ---------------------------------------------------------
    # Hand velocity
    # ---------------------------------------------------------
    velocity = np.zeros_like(hand)
    velocity[:, 1:, :] = hand[:, 1:, :] - hand[:, :-1, :]

    # ---------------------------------------------------------
    # Hand acceleration
    # ---------------------------------------------------------
    acceleration = np.zeros_like(hand)
    acceleration[:, 2:, :] = velocity[:, 2:, :] - velocity[:, 1:-1, :]

    # ---------------------------------------------------------
    # Velocity magnitude for each hand
    #
    # Left hand = landmarks 0:63
    # Right hand = landmarks 63:126
    # Each landmark has x,y,z
    # ---------------------------------------------------------
    left_velocity = velocity[:, :, :63].reshape(
        velocity.shape[0], velocity.shape[1], 21, 3
    )

    right_velocity = velocity[:, :, 63:126].reshape(
        velocity.shape[0], velocity.shape[1], 21, 3
    )

    left_speed = np.linalg.norm(left_velocity, axis=3).mean(axis=2)
    right_speed = np.linalg.norm(right_velocity, axis=3).mean(axis=2)

    hand_speed = np.stack(
        [left_speed, right_speed],
        axis=2
    )

    # ---------------------------------------------------------
    # Acceleration magnitude
    # ---------------------------------------------------------
    left_acc = acceleration[:, :, :63].reshape(
        acceleration.shape[0], acceleration.shape[1], 21, 3
    )

    right_acc = acceleration[:, :, 63:126].reshape(
        acceleration.shape[0], acceleration.shape[1], 21, 3
    )

    left_acc_mag = np.linalg.norm(left_acc, axis=3).mean(axis=2)
    right_acc_mag = np.linalg.norm(right_acc, axis=3).mean(axis=2)

    hand_acceleration = np.stack(
        [left_acc_mag, right_acc_mag],
        axis=2
    )

    # ---------------------------------------------------------
    # Optical-flow change
    # ---------------------------------------------------------
    flow_delta = np.zeros_like(flow_motion)
    flow_delta[:, 1:, :] = (
        flow_motion[:, 1:, :] -
        flow_motion[:, :-1, :]
    )

    # ---------------------------------------------------------
    # Combine
    #
    # 139 original
    # + 126 velocity
    # + 126 acceleration
    # + 2 hand speed
    # + 2 hand acceleration magnitude
    # + 1 flow change
    #
    # = 396 features/frame
    # ---------------------------------------------------------
    X_v2 = np.concatenate(
        [
            X,
            velocity,
            acceleration,
            hand_speed,
            hand_acceleration,
            flow_delta
        ],
        axis=2
    )

    return X_v2.astype(np.float32)


def process_file(input_file, output_file):

    data = np.load(input_file, allow_pickle=True)

    X = data["X"]
    y = data["y"]

    print("\nLoading:", input_file)
    print("Original shape:", X.shape)

    X_v2 = add_temporal_features(X)

    print("V2 shape:", X_v2.shape)

    np.savez_compressed(
        output_file,
        X=X_v2,
        y=y
    )

    print("Saved:", output_file)


def main():

    print("=" * 70)
    print("V4 ISL — TEMPORAL FEATURE PREPARATION V2")
    print("=" * 70)

    process_file(TRAIN_FILE, OUT_TRAIN)
    process_file(TEST_FILE, OUT_TEST)

    print("\n" + "=" * 70)
    print("V2 PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()