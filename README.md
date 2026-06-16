# ARKIST Motion AI

Smartphone-based motion tracking and feedback prototyping for AR-supported therapeutic exercise.

## Current Prototype

- MediaPipe Pose captures 33 3D body landmarks from a camera stream.
- Motion sequences can be stored as NumPy arrays for offline analysis.
- First rule-based feature extraction exists for therapeutic movement feedback.
- Exercise descriptions and feedback criteria are parsed from the demo Excel material into structured JSON.

## Near-Term Architecture

```text
Smartphone camera
  -> Pose tracking
  -> Tracking quality checks
  -> Pose normalization
  -> Repetition segmentation
  -> Exercise-specific feature extraction
  -> Rule-based safety feedback
  -> Later: unsupervised deviation detection
  -> AR/VR feedback interface
```

## Important Data Policy

This repository is prepared for code and structured, shareable metadata. Large files and private research assets are intentionally excluded from Git by default:

- recorded movement sequences (`.npy`, `.npz`)
- model weights and MediaPipe task files (`.pt`, `.task`, `.onnx`)
- papers and internal documents (`.pdf`, Office files)
- raw participant or patient data

Store sensitive or large artifacts outside Git, or add them later via a deliberate data management process.

## Presentation Focus

For the next project presentation, the strongest demonstrator is a professionalized motion-analysis pipeline:

- 3D skeleton tracking from smartphone camera input
- pseudo-body architecture with torso, limb axes, hip/shoulder width and floor reference
- tracking quality indicators
- therapeutic feature extraction
- first feedback generation

## Generate Body Architecture Preview

Use an existing MediaPipe sequence to render a presentation-ready pseudo body architecture image:

```bash
python mediapipe_service/body_architecture_visualizer.py mediapipe_service/test_squat_slow_frontal_02_30s.npy --output presentation_assets/body_architecture_preview.png
```

The generated image is kept out of Git by default because it is a reproducible presentation artifact.
