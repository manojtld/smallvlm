"""
Evaluation metrics for Level 1 (normal/abnormal) and Level 2 (per-label findings).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """Accuracy, sensitivity (recall), specificity, precision, F1 for binary classification."""
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy    = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "accuracy":    round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision":   round(precision, 4),
        "f1":          round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": len(y_true),
    }


def multilabel_metrics(
    y_true: Dict[str, List[bool]],
    y_pred: Dict[str, List[bool]],
    labels: List[str],
) -> Dict:
    """
    Per-label and macro-averaged metrics for multilabel classification.

    Args:
        y_true: {label: [bool, ...]} ground truth per label across all samples
        y_pred: {label: [bool, ...]} predictions per label across all samples
        labels: ordered list of label names
    """
    per_label = {}
    f1_scores = []

    for label in labels:
        gt = y_true.get(label, [])
        pred = y_pred.get(label, [])
        if not gt:
            continue
        m = binary_metrics(gt, pred)
        per_label[label] = m
        f1_scores.append(m["f1"])

    macro = {
        "macro_f1":          round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "macro_sensitivity": round(float(np.mean([per_label[l]["sensitivity"] for l in per_label])), 4),
        "macro_specificity": round(float(np.mean([per_label[l]["specificity"] for l in per_label])), 4),
        "macro_precision":   round(float(np.mean([per_label[l]["precision"] for l in per_label])), 4),
    }

    return {"per_label": per_label, "macro": macro}


def print_binary_results(name: str, metrics: Dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  N={metrics['n']}  accuracy={metrics['accuracy']:.3f}")
    print(f"  sensitivity={metrics['sensitivity']:.3f}  specificity={metrics['specificity']:.3f}")
    print(f"  precision={metrics['precision']:.3f}  F1={metrics['f1']:.3f}")
    print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")


def print_multilabel_results(results: Dict, labels: List[str]) -> None:
    macro = results["macro"]
    per_label = results["per_label"]
    print(f"\n{'='*70}")
    print(f"  Level 2 — Finding Presence/Absence")
    print(f"{'='*70}")
    print(f"  macro F1={macro['macro_f1']:.3f}  sens={macro['macro_sensitivity']:.3f}  "
          f"spec={macro['macro_specificity']:.3f}  prec={macro['macro_precision']:.3f}")
    print(f"\n  {'Label':35s}  {'F1':>6}  {'Sens':>6}  {'Spec':>6}  {'N+':>5}")
    print(f"  {'-'*65}")
    for label in labels:
        if label not in per_label:
            continue
        m = per_label[label]
        n_pos = m["tp"] + m["fn"]
        print(f"  {label:35s}  {m['f1']:6.3f}  {m['sensitivity']:6.3f}  {m['specificity']:6.3f}  {n_pos:5d}")
