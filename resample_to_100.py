import numpy as np

def resample_sequence(seq, target_len=100):
    old_len = seq.shape[0]
    old_idx = np.linspace(0, old_len - 1, old_len)
    new_idx = np.linspace(0, old_len - 1, target_len)

    flat = seq.reshape(old_len, -1)
    resampled = np.zeros((target_len, flat.shape[1]), dtype=np.float32)

    for i in range(flat.shape[1]):
        resampled[:, i] = np.interp(new_idx, old_idx, flat[:, i])

    return resampled.reshape(target_len, *seq.shape[1:])

pos = np.load("train_x_pos.npy")[0]
ori = np.load("train_x_ori.npy")[0]
y = np.load("train_y.npy")

pos_100 = resample_sequence(pos, 100)
ori_100 = resample_sequence(ori, 100)

pos_100 = np.expand_dims(pos_100, axis=0)
ori_100 = np.expand_dims(ori_100, axis=0)

np.save("train_x_pos_100.npy", pos_100)
np.save("train_x_ori_100.npy", ori_100)
np.save("train_y_100.npy", y)

print("Saved:")
print("train_x_pos_100.npy", pos_100.shape)
print("train_x_ori_100.npy", ori_100.shape)
print("train_y_100.npy", y.shape)