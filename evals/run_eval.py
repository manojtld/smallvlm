"""
Run the CXR evaluation benchmark on a model.

Usage:
  python -m evals.run_eval
  python -m evals.run_eval --device cuda:0 --output evals/results/baseline.json

Levels:
  1 — normal/abnormal classification
  2 — closed-vocab finding presence/absence (14 labels)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from .dataset import load_test_split
from .inference import QwenEvaluator
from .metrics import (
    binary_metrics, multilabel_metrics,
    print_binary_results, print_multilabel_results,
)
from .vocab import EVAL_LABELS

RESULTS_DIR = Path(__file__).parent / "results"


def run(device: str = "cuda:0", output: str = None) -> dict:
    samples = load_test_split()
    evaluator = QwenEvaluator(device=device)

    # ── Level 1: normal/abnormal ──────────────────────────────────────────────
    l1_true, l1_pred = [], []
    skipped_l1 = 0

    # ── Level 2: per-label presence ──────────────────────────────────────────
    l2_true = defaultdict(list)
    l2_pred = defaultdict(list)
    skipped_l2 = 0

    print(f"\nRunning eval on {len(samples)} samples...")
    for sample in tqdm(samples):
        image = sample.frontal_image

        # Level 1
        pred_normal = evaluator.predict_normal(image)
        if pred_normal is None:
            skipped_l1 += 1
        else:
            l1_true.append(sample.is_normal)
            l1_pred.append(pred_normal)

        # Level 2
        pred_labels = evaluator.predict_labels(image, EVAL_LABELS)
        has_any = any(v is not None for v in pred_labels.values())
        if not has_any:
            skipped_l2 += 1
        else:
            for label in EVAL_LABELS:
                gt = label in sample.problems
                pred = pred_labels.get(label)
                if pred is not None:
                    l2_true[label].append(gt)
                    l2_pred[label].append(pred)

    print(f"\nLevel 1 skipped (no image / unparseable): {skipped_l1}")
    print(f"Level 2 skipped (no image / unparseable): {skipped_l2}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    l1_metrics = binary_metrics(l1_true, l1_pred)
    l2_results = multilabel_metrics(dict(l2_true), dict(l2_pred), EVAL_LABELS)

    print_binary_results("Level 1 — Normal/Abnormal Classification", l1_metrics)
    print_multilabel_results(l2_results, EVAL_LABELS)

    results = {
        "model": "Qwen/Qwen3.5-0.8B",
        "n_samples": len(samples),
        "level1": l1_metrics,
        "level2": l2_results,
    }

    # Save
    out_path = Path(output) if output else RESULTS_DIR / "baseline_qwen35_08b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(device=args.device, output=args.output)


if __name__ == "__main__":
    main()
