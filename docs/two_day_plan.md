# Two-Day Plan for Presentation

Date context: Tuesday, 2026-06-16. Presentation target: Thursday, 2026-06-18.

## Goal

Show more than a raw tracking simulation: present a credible research prototype architecture for smartphone-based therapeutic motion assessment.

## Tuesday

1. Connect the project to GitHub with a safe repository structure.
2. Add a professional README and data protection defaults.
3. Stabilize the concept of the processing pipeline:
   `capture -> quality -> normalize -> segment -> features -> feedback`.
4. Inspect current `.npy` recordings and define what the demo should show.

## Wednesday

1. Build a stronger 3D visualization:
   - body skeleton
   - floor plane
   - torso plane
   - shoulder and hip width
   - leg axes and arm axes
2. Add basic tracking quality metrics:
   - person detected
   - landmark completeness
   - frame count
   - motion stability
3. Create an offline analysis entry point for saved movement files.
4. Export screenshots or short demo assets for the presentation.

## Thursday Morning

1. Run the demo once end to end.
2. Prepare talking points:
   - why smartphone-only tracking is valuable
   - why safety checks stay rule-based
   - where unsupervised learning enters later
   - how the AR/VR team can consume the feedback layer

