"""
Fixed evaluation vocabulary — the closed set of findings used for Level 2 eval.

Selected from top Problems tags in the IU dataset, filtered to clinically
meaningful pathological findings (excluding normal, pure anatomy, and noise tags).
"""

from __future__ import annotations

# 14 labels — balanced between common and less common findings, all pathological
EVAL_LABELS = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pulmonary Atelectasis",
    "Opacity",
    "Pulmonary Edema",
    "Pulmonary Congestion",
    "Emphysema",
    "Nodule",
    "Infiltrate",
    "Airspace Disease",
    "Calcinosis",
    "Calcified Granuloma",
    "Fractures, Bone",
    "Catheters, Indwelling",
]
