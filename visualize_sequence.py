import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

arr = np.load("test_squat_slow_frontal_02_30s.npy")

connections = [
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),
]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

def update(frame_idx):

    ax.clear()

    points = arr[frame_idx].copy()

    # Hüftzentrum berechnen
    foot_center = (points[29] + points[30]) / 2
    points = points - foot_center

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, z, -y, s=40)

    for a, b in connections:

        ax.plot(
            [x[a], x[b]],
            [z[a], z[b]],
            [-y[a], -y[b]]
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    ax.set_title(f"Frame {frame_idx}")

ani = FuncAnimation(
    fig,
    update,
    frames=len(arr),
    interval=50
)

plt.show()