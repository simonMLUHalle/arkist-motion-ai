import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
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


SKELETON_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
]


def load_sequence(path: Path) -> np.ndarray:
    sequence = np.load(path)
    if sequence.ndim != 3 or sequence.shape[1:] != (33, 3):
        raise ValueError(f"Expected shape (frames, 33, 3), got {sequence.shape}")
    return sequence.astype(np.float32)


def center_on_feet(points: np.ndarray) -> np.ndarray:
    foot_center = (points[LEFT_HEEL] + points[RIGHT_HEEL]) / 2.0
    return points - foot_center


def to_scene_axes(points: np.ndarray) -> np.ndarray:
    scene = np.zeros_like(points)
    scene[:, 0] = points[:, 0]
    scene[:, 1] = points[:, 2]
    scene[:, 2] = -points[:, 1]
    return scene


def select_representative_frame(sequence: np.ndarray) -> int:
    wrist_center = (sequence[:, LEFT_WRIST, :] + sequence[:, RIGHT_WRIST, :]) / 2.0
    vertical = wrist_center[:, 1]
    return int(np.argmax(vertical))


def distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def sequence_quality(sequence: np.ndarray) -> dict:
    finite_ratio = float(np.isfinite(sequence).mean())
    frame_count = int(sequence.shape[0])

    hip_center = (sequence[:, LEFT_HIP, :] + sequence[:, RIGHT_HIP, :]) / 2.0
    step_motion = np.linalg.norm(np.diff(hip_center, axis=0), axis=1)
    stability = float(np.percentile(step_motion, 95)) if len(step_motion) else 0.0

    if finite_ratio < 0.98:
        label = "check"
    elif stability > 0.08:
        label = "motion/noisy"
    else:
        label = "usable"

    return {
        "frames": frame_count,
        "finite_ratio": finite_ratio,
        "hip_motion_p95": stability,
        "label": label,
    }


def architecture_metrics(points: np.ndarray) -> dict:
    shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_width = distance(points, LEFT_HIP, RIGHT_HIP)
    heel_width = distance(points, LEFT_HEEL, RIGHT_HEEL)
    hand_asymmetry = abs(points[LEFT_WRIST, 1] - points[RIGHT_WRIST, 1])

    return {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "heel_to_hip_ratio": heel_width / max(hip_width, 1e-6),
        "hand_height_asymmetry": float(hand_asymmetry),
    }


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def project(scene_points: np.ndarray, width: int, height: int, scale: float = 650.0) -> np.ndarray:
    x = scene_points[:, 0]
    depth = scene_points[:, 1]
    z = scene_points[:, 2]

    px = width * 0.43 + scale * (x + 0.34 * depth)
    py = height * 0.83 - scale * (z - 0.20 * depth)
    return np.column_stack([px, py])


def draw_line(draw: ImageDraw.ImageDraw, projected: np.ndarray, a: int, b: int, color: str, width: int = 4) -> None:
    draw.line([tuple(projected[a]), tuple(projected[b])], fill=hex_to_rgb(color), width=width)


def draw_joint(draw: ImageDraw.ImageDraw, projected: np.ndarray, idx: int, color: str, radius: int = 8) -> None:
    x, y = projected[idx]
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=hex_to_rgb(color))


def draw_floor(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    floor = [(130, height - 120), (width - 250, height - 120), (width - 80, height - 230), (300, height - 230)]
    draw.polygon(floor, fill=(220, 238, 225), outline=(135, 178, 151))
    for i in range(6):
        t = i / 5
        x1 = floor[0][0] * (1 - t) + floor[1][0] * t
        y1 = floor[0][1] * (1 - t) + floor[1][1] * t
        x2 = floor[3][0] * (1 - t) + floor[2][0] * t
        y2 = floor[3][1] * (1 - t) + floor[2][1] * t
        draw.line([(x1, y1), (x2, y2)], fill=(185, 211, 195), width=1)
    for i in range(5):
        t = i / 4
        x1 = floor[0][0] * (1 - t) + floor[3][0] * t
        y1 = floor[0][1] * (1 - t) + floor[3][1] * t
        x2 = floor[1][0] * (1 - t) + floor[2][0] * t
        y2 = floor[1][1] * (1 - t) + floor[2][1] * t
        draw.line([(x1, y1), (x2, y2)], fill=(185, 211, 195), width=1)


def plot_body_architecture(sequence: np.ndarray, frame_idx: int, output: Path) -> None:
    raw_points = sequence[frame_idx]
    points = center_on_feet(raw_points)
    scene = to_scene_axes(points)
    width, height = 1800, 1200
    projected = project(scene, width, height)

    quality = sequence_quality(sequence)
    metrics = architecture_metrics(points)

    image = Image.new("RGB", (width, height), "#f7f7f2")
    draw = ImageDraw.Draw(image, "RGBA")

    draw_floor(draw, width, height)

    torso_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP]
    torso = [tuple(projected[i]) for i in torso_indices]
    draw.polygon(torso, fill=(59, 130, 246, 55), outline=(29, 78, 216, 210))

    for a, b in SKELETON_CONNECTIONS:
        draw_line(draw, projected, a, b, "#222222", 5)

    draw_line(draw, projected, LEFT_SHOULDER, RIGHT_SHOULDER, "#f97316", 9)
    draw_line(draw, projected, LEFT_HIP, RIGHT_HIP, "#14b8a6", 9)
    draw_line(draw, projected, LEFT_HIP, LEFT_ANKLE, "#dc2626", 7)
    draw_line(draw, projected, RIGHT_HIP, RIGHT_ANKLE, "#dc2626", 7)
    draw_line(draw, projected, LEFT_WRIST, RIGHT_WRIST, "#2563eb", 5)

    for idx in range(len(projected)):
        draw_joint(draw, projected, idx, "#111827", 6)

    for idx in [LEFT_WRIST, RIGHT_WRIST, LEFT_HEEL, RIGHT_HEEL, LEFT_KNEE, RIGHT_KNEE]:
        draw_joint(draw, projected, idx, "#2563eb", 11)

    try:
        title_font = ImageFont.truetype("Arial.ttf", 46)
        text_font = ImageFont.truetype("Arial.ttf", 28)
        small_font = ImageFont.truetype("Arial.ttf", 22)
    except OSError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((70, 52), "ARKIST Motion AI - Pseudo Body Architecture", fill="#111827", font=title_font)
    draw.text((74, 112), "3D MediaPipe landmarks -> body axes, torso plane, floor reference, quality metrics", fill="#3f3f46", font=small_font)

    info = (
        f"frame: {frame_idx}/{len(sequence) - 1}\n"
        f"tracking: {quality['label']} | frames: {quality['frames']}\n"
        f"shoulder width: {metrics['shoulder_width']:.2f} m\n"
        f"hip width: {metrics['hip_width']:.2f} m\n"
        f"foot/hip ratio: {metrics['heel_to_hip_ratio']:.2f}\n"
        f"hand asymmetry: {metrics['hand_height_asymmetry']:.2f} m"
    )
    panel = (1260, 270, 1725, 575)
    draw.rounded_rectangle(panel, radius=18, fill=(255, 255, 255, 235), outline=(210, 210, 200, 255), width=2)
    draw.text((1300, 310), info, fill="#111827", font=text_font, spacing=10)

    legend_items = [
        ("#3b82f6", "torso surface"),
        ("#f97316", "shoulder axis"),
        ("#14b8a6", "hip axis"),
        ("#dc2626", "leg axis"),
        ("#2563eb", "tracked endpoints"),
    ]
    y0 = 640
    for color, label in legend_items:
        draw.rectangle((1300, y0, 1340, y0 + 20), fill=hex_to_rgb(color))
        draw.text((1360, y0 - 6), label, fill="#111827", font=small_font)
        y0 += 48

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an ARKIST body architecture preview from a MediaPipe sequence.")
    parser.add_argument("input", type=Path, help="Path to a MediaPipe .npy sequence with shape (frames, 33, 3).")
    parser.add_argument("--output", type=Path, default=Path("presentation_assets/body_architecture_preview.png"))
    parser.add_argument("--frame", type=int, default=None, help="Frame index. Defaults to the lowest-hand representative frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = load_sequence(args.input)
    frame_idx = args.frame if args.frame is not None else select_representative_frame(sequence)
    frame_idx = max(0, min(frame_idx, len(sequence) - 1))
    plot_body_architecture(sequence, frame_idx, args.output)
    print(f"Saved body architecture preview to {args.output}")


if __name__ == "__main__":
    main()
