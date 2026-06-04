from fastapi import FastAPI
import uvicorn
import cv2
import threading
import math
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()

latest_status = {
    "has_person": 0,
    "num_landmarks": 0,
    "num_world_landmarks": 0,
    "left_knee_angle": None,
    "right_knee_angle": None,
    "avg_knee_angle": None,
}

# =========================================
# Sequence Buffer for Temporal 3D Skeletons
# =========================================

sequence_buffer = []
MAX_SEQUENCE_LENGTH = 1200

# =========================================
# MediaPipe Pose Landmarker Setup
# =========================================

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# =========================================
# Camera Setup
# =========================================

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# =========================================
# Angle Calculation
# =========================================

def angle_2d(a, b, c):

    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y

    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    dot = v1[0] * v2[0] + v1[1] * v2[1]

    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if n1 == 0 or n2 == 0:
        return None

    cosang = max(-1, min(1, dot / (n1 * n2)))

    return math.degrees(math.acos(cosang))

# =========================================
# Camera Loop
# =========================================

def camera_loop():

    global latest_status
    global sequence_buffer

    timestamp_ms = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            lm = result.pose_landmarks[0]

            # =========================================
            # 3D WORLD LANDMARK STORAGE
            # =========================================

            if result.pose_world_landmarks:

                world_lm = result.pose_world_landmarks[0]

                frame_3d = []

                for p in world_lm:
                    frame_3d.append([
                        p.x,
                        p.y,
                        p.z
                    ])

                sequence_buffer.append(frame_3d)

                if len(sequence_buffer) > MAX_SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)

            # =========================================
            # Knee Angles
            # =========================================

            left_knee = angle_2d(
                lm[23],
                lm[25],
                lm[27]
            )

            right_knee = angle_2d(
                lm[24],
                lm[26],
                lm[28]
            )

            vals = [
                v for v in [left_knee, right_knee]
                if v is not None
            ]

            avg = sum(vals) / len(vals) if vals else None

            latest_status = {
                "has_person": 1,
                "num_landmarks": len(lm),
                "num_world_landmarks": len(result.pose_world_landmarks[0])
                if result.pose_world_landmarks else 0,
                "left_knee_angle": left_knee,
                "right_knee_angle": right_knee,
                "avg_knee_angle": avg,
            }

            # =========================================
            # Draw Skeleton Points
            # =========================================

            h, w, _ = frame.shape

            for point in lm:

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

        else:

            latest_status = {
                "has_person": 0,
                "num_landmarks": 0,
                "num_world_landmarks": 0,
                "left_knee_angle": None,
                "right_knee_angle": None,
                "avg_knee_angle": None,
            }

        # =========================================
        # Debug Output
        # =========================================

        print(
            f"Person: {latest_status['has_person']} | "
            f"Landmarks: {latest_status['num_landmarks']} | "
            f"World: {latest_status['num_world_landmarks']} | "
            f"Knee Angle: {latest_status['avg_knee_angle']} | "
            f"Sequence Frames: {len(sequence_buffer)}"
        )

        timestamp_ms += 33

# =========================================
# API ENDPOINTS
# =========================================

@app.get("/status")
def get_status():

    return latest_status

@app.get("/sequence")
def get_sequence():

    return {
        "frames": len(sequence_buffer),
        "sequence": sequence_buffer
    }

@app.get("/save_sequence")
def save_sequence():

    arr = np.array(
        sequence_buffer,
        dtype=np.float32
    )

    timestamp = int(time.time())

    filename = f"sequence_{timestamp}.npy"

    np.save(filename, arr)

    return {
        "saved": True,
        "filename": filename,
        "shape": arr.shape
    }

# =========================================
# Start Background Thread
# =========================================

threading.Thread(
    target=camera_loop,
    daemon=True
).start()

# =========================================
# Run Server
# =========================================

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8001
)
