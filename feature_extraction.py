import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).parent

SEQUENCE_FILE = BASE_DIR / "mediapipe_service/test_squat_slow_frontal_02_30s.npy"


# MediaPipe Landmark Indices
NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

LEFT_HEEL = 29
RIGHT_HEEL = 30

LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LEFT_WRIST = 15
RIGHT_WRIST = 16


def load_sequence(path=SEQUENCE_FILE):
    arr = np.load(path)
    print("Loaded sequence:", arr.shape)
    return arr


def center_between(points, idx_a, idx_b):
    return (points[:, idx_a, :] + points[:, idx_b, :]) / 2.0


def vertical_distance(a, b):
    return np.abs(a[:, 1] - b[:, 1])


def hand_knee_relation(seq):
    """
    Misst, ob Hände über/unter Kniehöhe sind.
    Kleinere y-Werte = weiter oben, größere y-Werte = weiter unten.
    """
    hand_center = center_between(seq, LEFT_WRIST, RIGHT_WRIST)
    knee_center = center_between(seq, LEFT_KNEE, RIGHT_KNEE)
    ankle_center = center_between(seq, LEFT_ANKLE, RIGHT_ANKLE)

    hand_y = hand_center[:, 1]
    knee_y = knee_center[:, 1]
    ankle_y = ankle_center[:, 1]

    lowest_frame = np.argmax(hand_y)

    if hand_y[lowest_frame] < knee_y[lowest_frame]:
        level = "hands_above_knees"
    elif hand_y[lowest_frame] < ankle_y[lowest_frame]:
        level = "hands_between_knees_and_ankles"
    else:
        level = "hands_below_ankles_or_near_floor"

    return {
        "lowest_frame": int(lowest_frame),
        "level": level,
        "hand_y": float(hand_y[lowest_frame]),
        "knee_y": float(knee_y[lowest_frame]),
        "ankle_y": float(ankle_y[lowest_frame]),
    }


def hand_height_asymmetry(seq):
    left_y = seq[:, LEFT_WRIST, 1]
    right_y = seq[:, RIGHT_WRIST, 1]

    asym = np.abs(left_y - right_y)

    return {
        "max_asymmetry": float(np.max(asym)),
        "mean_asymmetry": float(np.mean(asym)),
        "worst_frame": int(np.argmax(asym)),
    }


def foot_width(seq):
    left_heel = seq[:, LEFT_HEEL, :]
    right_heel = seq[:, RIGHT_HEEL, :]
    left_hip = seq[:, LEFT_HIP, :]
    right_hip = seq[:, RIGHT_HIP, :]

    heel_width = np.linalg.norm(left_heel - right_heel, axis=1)
    hip_width = np.linalg.norm(left_hip - right_hip, axis=1)

    ratio = heel_width / np.maximum(hip_width, 1e-6)

    return {
        "mean_foot_hip_ratio": float(np.mean(ratio)),
        "min_foot_hip_ratio": float(np.min(ratio)),
        "max_foot_hip_ratio": float(np.max(ratio)),
    }


def movement_speed(seq):
    wrist_center = center_between(seq, LEFT_WRIST, RIGHT_WRIST)

    diffs = np.diff(wrist_center, axis=0)
    speed = np.linalg.norm(diffs, axis=1)

    return {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "peak_speed_frame": int(np.argmax(speed)),
    }


def analyze_jefferson_curl(seq):
    return {
        "exercise": "Jefferson Curl",
        "hand_knee_relation": hand_knee_relation(seq),
        "hand_height_asymmetry": hand_height_asymmetry(seq),
        "foot_width": foot_width(seq),
        "movement_speed": movement_speed(seq),
    }


if __name__ == "__main__":
    seq = load_sequence()
    result = analyze_jefferson_curl(seq)

    print()
    print("Analysis result:")
    print(result)