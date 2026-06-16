from dataclasses import dataclass
from typing import Optional
import sys
import time
import threading

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO


# ============================
#  Konfiguration & Konstanten
# ============================

@dataclass
class SquatConfig:
    knee_valgus_good: float = 0.05
    knee_valgus_warning: float = 0.10
    knee_valgus_abort: float = 0.15
    depth_target_angle: float = 90.0
    depth_tolerance_deg: float = 20.0
    min_keypoint_conf: float = 0.5


# Integer-Codes aligned with Unity (0=Critical, 1=Warning, 2=Good)
# -1 = Unknown (wird in C# übersprungen)

# Knievalgus / "relativer Abstand Knie"
KNEE_VALGUS_UNKNOWN = -1   # Tracking zu unsicher
KNEE_VALGUS_CRITICAL = 0   # > ~15 cm → Critical (Abbruch!)
KNEE_VALGUS_WARNING = 1    # ~10–15 cm → Warning
KNEE_VALGUS_GOOD = 2       # < ~5 cm → Good

# Tiefe / Kniewinkel
DEPTH_UNKNOWN = -1         # Tracking zu unsicher
DEPTH_CRITICAL = 0         # sehr tief → Critical (Vorsicht)
DEPTH_WARNING = 1          # zu wenig tief → Warning 
DEPTH_GOOD = 2             # ca. 90° ± Toleranz → Good

# Hüfte vs. Kniehöhe
HIP_KNEE_UNKNOWN = -1      # Tracking zu unsicher
HIP_KNEE_CRITICAL = 0      # sehr tief → Critical (Vorsicht)
HIP_KNEE_WARNING = 1       # Hüfte über Knie → Warning (tiefer gehen)
HIP_KNEE_GOOD = 2          # Hüfte auf Kniehöhe → Good

# Arme
ARMS_UNKNOWN = -1          # Arme nicht erkennbar
ARMS_WARNING = 1           # nicht korrekt → Warning (korrigieren)
ARMS_GOOD = 2              # korrekt → Good


# Abtastrate: nur alle 0.1 Sekunden neue Analyse
MIN_INTERVAL_SECONDS = 0.1
CAMERA_INDEX = 0


# ============================
#  Geometrie-Hilfsfunktionen
# ============================

def calculate_angle(a, b, c):
    """Winkel an Punkt b (Grad) zwischen ba und bc."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return None
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def knee_valgus_metric(hip, knee, ankle):
    """Abstand des Knies von der Linie Hüfte–Sprunggelenk, normiert auf Beinlänge."""
    hip = np.array(hip, dtype=float)
    knee = np.array(knee, dtype=float)
    ankle = np.array(ankle, dtype=float)

    leg_vec = ankle - hip
    seg_vec = knee - hip
    leg_len = np.linalg.norm(leg_vec)
    if leg_len < 1e-6:
        return None

    cross_val = abs(np.cross(leg_vec, seg_vec)) / leg_len
    return cross_val / leg_len


# ============================
#  SquatAnalyzer
# ============================

class SquatAnalyzer:
    """Analysiert EIN Frame und gibt Integer-States zurück."""

    def __init__(self, cfg: SquatConfig):
        self.cfg = cfg

    def _check_arms(self, left_shoulder, right_shoulder, left_wrist, right_wrist, visible_mask) -> int:
        """Prüft Armposition parallel zum Boden auf Schulterhöhe."""

        def arm_ok(shoulder, wrist) -> bool:
            v = np.array(wrist, dtype=float) - np.array(shoulder, dtype=float)
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                return False

            dx, dy = v[0], v[1]
            # y-Achse im Bild zeigt nach unten -> invertieren für Winkel-Berechnung
            angle = np.degrees(np.arctan2(-dy, dx))  # 0°: nach rechts, 180°: nach links
            horiz_dev = min(abs(angle), abs(abs(angle) - 180.0))
            if horiz_dev > 20.0:
                return False

            # Höhe Hand ungefähr Schulterhöhe (normiert auf Schulterbreite)
            shoulder_width = np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder))
            if shoulder_width < 1e-6:
                return False
            height_diff_norm = abs(dy) / shoulder_width
            if height_diff_norm > 0.3:
                return False

            return True

        if not visible_mask["L_SHOULDER"] or not visible_mask["R_SHOULDER"]:
            return ARMS_UNKNOWN
        if not visible_mask["L_WRIST"] and not visible_mask["R_WRIST"]:
            return ARMS_UNKNOWN

        left_ok = visible_mask["L_WRIST"] and arm_ok(left_shoulder, left_wrist)
        right_ok = visible_mask["R_WRIST"] and arm_ok(right_shoulder, right_wrist)
        if left_ok or right_ok:
            return ARMS_GOOD
        return ARMS_WARNING

    def analyze_frame(self, kpts: np.ndarray, image_shape):
        """Analysiert Frame und gibt States zurück."""
        H, W = image_shape[:2]

        # COCO keypoint indices
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16
        L_WRIST, R_WRIST = 9, 10

        def pt(idx):
            return kpts[idx, :2]

        def conf(idx):
            return kpts[idx, 2]

        visible_mask = {
            "L_SHOULDER": conf(L_SHOULDER) >= self.cfg.min_keypoint_conf,
            "R_SHOULDER": conf(R_SHOULDER) >= self.cfg.min_keypoint_conf,
            "L_HIP":      conf(L_HIP)      >= self.cfg.min_keypoint_conf,
            "R_HIP":      conf(R_HIP)      >= self.cfg.min_keypoint_conf,
            "L_KNEE":     conf(L_KNEE)     >= self.cfg.min_keypoint_conf,
            "R_KNEE":     conf(R_KNEE)     >= self.cfg.min_keypoint_conf,
            "L_ANKLE":    conf(L_ANKLE)    >= self.cfg.min_keypoint_conf,
            "R_ANKLE":    conf(R_ANKLE)    >= self.cfg.min_keypoint_conf,
            "L_WRIST":    conf(L_WRIST)    >= self.cfg.min_keypoint_conf,
            "R_WRIST":    conf(R_WRIST)    >= self.cfg.min_keypoint_conf,
        }

        critical = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]
        if not all(visible_mask[k] for k in critical):
            # Return UNKNOWN if tracking uncertain
            return {
                "knee_valgus_state": KNEE_VALGUS_UNKNOWN,
                "depth_state":       DEPTH_UNKNOWN,
                "hip_knee_state":    HIP_KNEE_UNKNOWN,
                "arms_state":        ARMS_UNKNOWN,
                "knee_valgus_metric": None,
                "avg_knee_angle":     None,
                "hip_knee_diff_norm": None,
            }

        # Calculate knee angles
        left_knee_angle = calculate_angle(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE))
        right_knee_angle = calculate_angle(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE))
        avg_knee_angle = None
        if left_knee_angle is not None and right_knee_angle is not None:
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        # Calculate knee valgus metric
        left_valgus = knee_valgus_metric(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE))
        right_valgus = knee_valgus_metric(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE))
        valgus_metric = None
        if left_valgus is not None and right_valgus is not None:
            valgus_metric = max(left_valgus, right_valgus)

        # Calculate hip vs knee height
        hip_center_y = (pt(L_HIP)[1] + pt(R_HIP)[1]) / 2.0
        knee_center_y = (pt(L_KNEE)[1] + pt(R_KNEE)[1]) / 2.0
        hip_knee_diff_norm = abs(hip_center_y - knee_center_y) / H

        # Evaluate knee valgus state
        if valgus_metric is None:
            knee_state = KNEE_VALGUS_UNKNOWN
        elif valgus_metric > self.cfg.knee_valgus_abort:
            knee_state = KNEE_VALGUS_CRITICAL  # > 15 cm
        elif valgus_metric > self.cfg.knee_valgus_warning:
            knee_state = KNEE_VALGUS_WARNING   # 10-15 cm
        elif valgus_metric < self.cfg.knee_valgus_good:
            knee_state = KNEE_VALGUS_GOOD      # < 5 cm
        else:
            knee_state = KNEE_VALGUS_WARNING   # mittlerer Bereich

        # Evaluate depth state
        if avg_knee_angle is None:
            depth_state = DEPTH_UNKNOWN
        else:
            target = self.cfg.depth_target_angle
            tol = self.cfg.depth_tolerance_deg
            if avg_knee_angle > target + tol:
                depth_state = DEPTH_WARNING     # zu wenig tief
            elif avg_knee_angle < target - tol:
                depth_state = DEPTH_CRITICAL    # sehr tief (Vorsicht)
            else:
                depth_state = DEPTH_GOOD        # ca. 90°

        # Evaluate hip-knee alignment
        if hip_knee_diff_norm is None:
            hip_state = HIP_KNEE_UNKNOWN
        else:
            # During squat, hips and knees will naturally be at different heights
            # We evaluate based on depth safety and form
            if hip_center_y < knee_center_y:
                # Hips are ABOVE knees in image coordinates (y increases downward)
                # Person is not squatting deep enough
                if hip_knee_diff_norm > 0.20:  # More than 20% = very shallow
                    hip_state = HIP_KNEE_WARNING
                else:
                    hip_state = HIP_KNEE_GOOD  # Acceptable depth (0-20%)
            else:
                # Hips are AT or BELOW knee level
                # Check if it's a safe deep squat or dangerously too deep
                if hip_knee_diff_norm > 0.30:  # Hips MORE than 30% below knees
                    hip_state = HIP_KNEE_CRITICAL  # ✅ TOO DEEP - Risk of injury!
                else:
                    hip_state = HIP_KNEE_GOOD  # Deep squat but still safe (0-30% below)

        # Evaluate arm position
        arms_state = self._check_arms(
            pt(L_SHOULDER), pt(R_SHOULDER), pt(L_WRIST), pt(R_WRIST), visible_mask
        )

        return {
            "knee_valgus_state":  knee_state,
            "depth_state":        depth_state,
            "hip_knee_state":     hip_state,
            "arms_state":         arms_state,
            "knee_valgus_metric": valgus_metric,
            "avg_knee_angle":     avg_knee_angle,
            "hip_knee_diff_norm": hip_knee_diff_norm,
        }


# ============================
#  FastAPI Models
# ============================

class SquatStatus(BaseModel):
    knee_valgus_state: int
    depth_state: int
    hip_knee_state: int
    arms_state: int
    has_person: int
    timestamp: float
    elapsed_since_update: float
    knee_valgus_metric: Optional[float]
    avg_knee_angle: Optional[float]
    hip_knee_diff_norm: Optional[float]


# ============================
#  FastAPI Setup
# ============================

app = FastAPI(title="YOLO11 Squat Service")

cfg = SquatConfig()

# Device detection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# YOLO11 Pose-Modell laden (Modellgröße bei Bedarf anpassen: n/s/m/l/x)
model = YOLO("yolo11n-pose.pt")
model.to(device)

analyzer = SquatAnalyzer(cfg)

# Global state
_last_analysis_time: float = 0.0
_latest_analysis: Optional[dict] = None


# ============================
#  Camera Loop (Always Running)
# ============================

def open_camera():
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Kamera konnte nicht geöffnet werden.")
    return cap


def camera_loop():
    """Camera runs continuously, analyzes every 2 seconds."""
    global _last_analysis_time, _latest_analysis
    
    cap = open_camera()
    print("Camera opened and ready.")
    
    while True:
        now = time.time()
        
        # Analyze every MIN_INTERVAL_SECONDS
        if now - _last_analysis_time >= MIN_INTERVAL_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Warning: No frame from camera.")
                time.sleep(0.1)
                continue

            # Run YOLO pose detection
            results = model.predict(frame, imgsz=640, conf=0.5, verbose=False, device=device)
            result = results[0]

            # 🆕 IMPROVED PERSON DETECTION LOGIC
            if result.keypoints is None or len(result.keypoints) == 0:
                # No keypoints detected at all
                print("[DEBUG] No keypoints detected - no person")
                _latest_analysis = None
            else:
                # Check if we have valid keypoints with sufficient confidence
                kpts_tensor = result.keypoints.data[0].cpu().numpy()
                
                # 🆕 CRITICAL FIX: Check if critical body parts are visible
                # COCO keypoint indices
                L_HIP, R_HIP = 11, 12
                L_KNEE, R_KNEE = 13, 14
                L_ANKLE, R_ANKLE = 15, 16
                
                # Check confidence of critical keypoints
                critical_indices = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
                min_conf_threshold = 0.3  # Lower threshold for person detection
                
                critical_keypoints_visible = sum(
                    1 for idx in critical_indices 
                    if idx < len(kpts_tensor) and kpts_tensor[idx, 2] >= min_conf_threshold
                )
                
                # Need ALL 6 critical keypoints to consider person detected
                if critical_keypoints_visible >= 6:  # ✅ CHANGED from 4 to 6
                    print(f"[DEBUG] Person detected - {critical_keypoints_visible}/6 critical keypoints visible")
                    _latest_analysis = analyzer.analyze_frame(kpts_tensor, frame.shape)
                else:
                    print(f"[DEBUG] Insufficient keypoints - only {critical_keypoints_visible}/6 visible - no person")
                    _latest_analysis = None
            
            _last_analysis_time = now
        else:
            time.sleep(0.05)


# Start camera thread on app startup
@app.on_event("startup")
def start_background_thread():
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()


# ============================
#  REST Endpoint
# ============================

@app.get("/status", response_model=SquatStatus)
def get_status():
    """Returns latest analysis or UNKNOWN states if no person detected."""
    now = time.time()
    
    # No analysis done yet
    if _last_analysis_time == 0.0:
        return SquatStatus(
            knee_valgus_state=KNEE_VALGUS_UNKNOWN,
            depth_state=DEPTH_UNKNOWN,
            hip_knee_state=HIP_KNEE_UNKNOWN,
            arms_state=ARMS_UNKNOWN,
            has_person=0,
            timestamp=0.0,
            elapsed_since_update=0.0,
            knee_valgus_metric=None,
            avg_knee_angle=None,
            hip_knee_diff_norm=None,
        )

    elapsed = now - _last_analysis_time

    # No person detected in last analysis
    if _latest_analysis is None:
        print("[STATUS] Returning: has_person=0, all states=UNKNOWN")
        return SquatStatus(
            knee_valgus_state=KNEE_VALGUS_UNKNOWN,
            depth_state=DEPTH_UNKNOWN,
            hip_knee_state=HIP_KNEE_UNKNOWN,
            arms_state=ARMS_UNKNOWN,
            has_person=0,
            timestamp=_last_analysis_time,
            elapsed_since_update=elapsed,
            knee_valgus_metric=None,
            avg_knee_angle=None,
            hip_knee_diff_norm=None,
        )

    # ✅ NEW: Debug output showing all states being returned
    print(f"[STATUS] Returning to server:")
    print(f"  ├─ has_person: 1")
    print(f"  ├─ KneeValgus: {_latest_analysis['knee_valgus_state']} ({get_state_name('knee_valgus', _latest_analysis['knee_valgus_state'])})")
    print(f"  ├─ Depth: {_latest_analysis['depth_state']} ({get_state_name('depth', _latest_analysis['depth_state'])})")
    print(f"  ├─ HipKnee: {_latest_analysis['hip_knee_state']} ({get_state_name('hip_knee', _latest_analysis['hip_knee_state'])})")
    print(f"  └─ Arms: {_latest_analysis['arms_state']} ({get_state_name('arms', _latest_analysis['arms_state'])})")

    # Return latest analysis
    return SquatStatus(
        knee_valgus_state=_latest_analysis["knee_valgus_state"],
        depth_state=_latest_analysis["depth_state"],
        hip_knee_state=_latest_analysis["hip_knee_state"],
        arms_state=_latest_analysis["arms_state"],
        has_person=1,
        timestamp=_last_analysis_time,
        elapsed_since_update=elapsed,
        knee_valgus_metric=_latest_analysis["knee_valgus_metric"],
        avg_knee_angle=_latest_analysis["avg_knee_angle"],
        hip_knee_diff_norm=_latest_analysis["hip_knee_diff_norm"],
    )


# ✅ NEW: Helper function for readable state names
def get_state_name(criterion: str, state: int) -> str:
    """Returns human-readable state name for debugging."""
    state_names = {
        -1: "UNKNOWN",
        0: "CRITICAL",
        1: "WARNING",
        2: "GOOD"
    }
    return state_names.get(state, f"INVALID({state})")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_squat_service:app", host="0.0.0.0", port=8000, reload=False)
