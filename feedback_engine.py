from feature_extraction import load_sequence, analyze_jefferson_curl


def evaluate_jefferson_curl(analysis):
    feedback = []
    warnings = []
    aborts = []

    hand_level = analysis["hand_knee_relation"]["level"]

    if hand_level == "hands_above_knees":
        feedback.append("Versuche mit den Händen noch etwas tiefer zu kommen.")
    elif hand_level == "hands_between_knees_and_ankles":
        feedback.append("Sehr gut. Wenn du dich wohl fühlst, kannst du versuchen, noch etwas tiefer zu kommen.")
    elif hand_level == "hands_below_ankles_or_near_floor":
        feedback.append("Sehr gut. Du erreichst einen großen Bewegungsumfang.")

    asym = analysis["hand_height_asymmetry"]["max_asymmetry"]

    if asym > 0.15:
        aborts.append("Abbruchkriterium: deutliche Asymmetrie der Handhöhe erkannt.")
    elif asym > 0.08:
        warnings.append("Achte darauf, beide Hände gleichmäßig nach unten zu führen.")

    foot_ratio = analysis["foot_width"]["mean_foot_hip_ratio"]

    if foot_ratio < 0.7:
        warnings.append("Stelle die Füße etwas weiter auseinander, ungefähr hüftbreit.")
    elif foot_ratio > 1.6:
        warnings.append("Stelle die Füße etwas näher zusammen, ungefähr hüftbreit.")
    else:
        feedback.append("Fußstellung wirkt plausibel hüft- bis schulterbreit.")

    speed = analysis["movement_speed"]["max_speed"]

    if speed > 0.12:
        warnings.append("Führe die Bewegung langsamer und kontrollierter aus.")
    else:
        feedback.append("Bewegungsgeschwindigkeit wirkt kontrolliert.")

    status = "good"

    if warnings:
        status = "warning"

    if aborts:
        status = "abort"

    return {
        "exercise": "Jefferson Curl",
        "status": status,
        "feedback": feedback,
        "warnings": warnings,
        "aborts": aborts,
        "raw_analysis": analysis
    }


if __name__ == "__main__":
    seq = load_sequence()
    analysis = analyze_jefferson_curl(seq)
    result = evaluate_jefferson_curl(analysis)

    print()
    print("Feedback result:")
    print("Status:", result["status"])

    print("\nFeedback:")
    for item in result["feedback"]:
        print("-", item)

    print("\nWarnings:")
    for item in result["warnings"]:
        print("-", item)

    print("\nAborts:")
    for item in result["aborts"]:
        print("-", item)