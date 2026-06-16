import numpy as np

# =========================================
# LOAD MEDIAPIPE SEQUENCE
# =========================================

arr = np.load("test_squat_slow_frontal_02_30s.npy")

print("Original shape:", arr.shape)

# =========================================
# SELECT IMPORTANT JOINTS
# =========================================

# MediaPipe indices
# hips, knees, ankles, shoulders, elbows, wrists

selected = [
    11, 12,   # shoulders
    13, 14,   # elbows
    15, 16,   # wrists

    23, 24,   # hips
    25, 26,   # knees
    27, 28    # ankles
]

data = arr[:, selected, :]

print("Reduced shape:", data.shape)

# =========================================
# NORMALIZE TO HIP CENTER
# =========================================

left_hip_idx = 6
right_hip_idx = 7

hip_center = (
    data[:, left_hip_idx, :] +
    data[:, right_hip_idx, :]
) / 2

data = data - hip_center[:, np.newaxis, :]

# =========================================
# POSITION STREAM
# =========================================

train_x_pos = data.copy()

# =========================================
# ORIENTATION STREAM
# =========================================

bones = [
    (0, 2),
    (2, 4),

    (1, 3),
    (3, 5),

    (6, 8),
    (8, 10),

    (7, 9),
    (9, 11),
]

orientation_frames = []

for frame in train_x_pos:

    frame_oris = []

    for a, b in bones:

        vec = frame[b] - frame[a]

        norm = np.linalg.norm(vec)

        if norm > 1e-6:
            vec = vec / norm

        frame_oris.append(vec)

    orientation_frames.append(frame_oris)

train_x_ori = np.array(orientation_frames)

# =========================================
# ADD BATCH DIMENSION
# =========================================

train_x_pos = np.expand_dims(train_x_pos, axis=0)
train_x_ori = np.expand_dims(train_x_ori, axis=0)

# Dummy label
train_y = np.array([[1]], dtype=np.float32)

# =========================================
# SAVE
# =========================================

np.save("train_x_pos.npy", train_x_pos)
np.save("train_x_ori.npy", train_x_ori)
np.save("train_y.npy", train_y)

print()
print("Saved:")
print("train_x_pos.npy", train_x_pos.shape)
print("train_x_ori.npy", train_x_ori.shape)
print("train_y.npy", train_y.shape)