import numpy as np

def resample_sequence(seq, target_len=100):
    old_len = seq.shape[0]
    old_idx = np.linspace(0, old_len - 1, old_len)
    new_idx = np.linspace(0, old_len - 1, target_len)

    flat = seq.reshape(old_len, -1)
    out = np.zeros((target_len, flat.shape[1]), dtype=np.float32)

    for i in range(flat.shape[1]):
        out[:, i] = np.interp(new_idx, old_idx, flat[:, i])

    return out.reshape(target_len, *seq.shape[1:])

arr = np.load("test_squat_slow_frontal_02_30s.npy")
arr100 = resample_sequence(arr, 100)

# 25 Kinect/Kuang-like joints from MediaPipe.
# Missing Kinect joints are approximated.
mp_map = [
    ("Spine_Base", 23, 24),
    ("Spine_Mid", 11, 12, 23, 24),
    ("Neck", 11, 12),
    ("Head", 0),

    ("Shoulder_Left", 11),
    ("Elbow_Left", 13),
    ("Wrist_Left", 15),
    ("Hand_Left", 19),

    ("Shoulder_Right", 12),
    ("Elbow_Right", 14),
    ("Wrist_Right", 16),
    ("Hand_Right", 20),

    ("Hip_Left", 23),
    ("Knee_Left", 25),
    ("Ankle_Left", 27),
    ("Foot_Left", 31),

    ("Hip_Right", 24),
    ("Knee_Right", 26),
    ("Ankle_Right", 28),
    ("Foot_Right", 32),

    ("Spine_Shoulder", 11, 12),
    ("Tip_Left", 19),
    ("Thumb_Left", 21),
    ("Tip_Right", 20),
    ("Thumb_Right", 22),
]

pos = np.zeros((100, 25, 3), dtype=np.float32)

for j, item in enumerate(mp_map):
    indices = item[1:]
    pts = arr100[:, indices, :]
    pos[:, j, :] = pts.mean(axis=1)

# Normalize around pelvis/root
root = pos[:, 0:1, :]
pos = pos - root

# =========================================================
# ORIENTATION STREAM
# =========================================================

parent = [
    0, 0, 1, 2,
    20, 4, 5, 6,
    20, 8, 9, 10,
    0, 12, 13, 14,
    0, 16, 17, 18,
    1, 7, 7, 11, 11
]

ori = np.zeros((100, 25, 4), dtype=np.float32)

for j in range(25):

    if j == parent[j]:
        continue

    vec = pos[:, j, :] - pos[:, parent[j], :]

    norm = np.linalg.norm(vec, axis=1, keepdims=True)

    vec = np.divide(
        vec,
        norm,
        out=np.zeros_like(vec),
        where=norm > 1e-6
    )

    ori[:, j, :3] = vec

# dummy quaternion w
ori[:, :, 3] = 0.0

# =========================================================
# FINAL SHAPES
# =========================================================

train_x_pos = np.expand_dims(pos, axis=0)
train_x_ori = np.expand_dims(ori, axis=0)
train_y = np.array([[1]], dtype=np.float32)

# =========================================================
# SAVE
# =========================================================

np.save("train_x_pos_25.npy", train_x_pos)
np.save("train_x_ori_25.npy", train_x_ori)
np.save("train_y_25.npy", train_y)

print()
print("SAVED")
print()

print("train_x_pos_25.npy:", train_x_pos.shape)
print("train_x_ori_25.npy:", train_x_ori.shape)
print("train_y_25.npy:", train_y.shape)