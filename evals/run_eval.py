"""
Run the CXR evaluation benchmark on a model.

Usage:
  python -m evals.run_eval
  python -m evals.run_eval --device cuda:0 --output evals/results/baseline.json

Outputs (both always written):
  <output>            — aggregated metrics JSON
  <output>.traces.jsonl — per-sample traces with uid, image, ground truth,
                          raw model response, and parsed prediction for every sample
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

    out_path = Path(output) if output else RESULTS_DIR / "baseline_qwen35_08b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path = out_path.with_suffix(".traces.jsonl")

    l1_true, l1_pred = [], []
    l2_true = defaultdict(list)
    l2_pred = defaultdict(list)
    skipped_l1 = skipped_l2 = 0

    print(f"\nRunning eval on {len(samples)} samples...")
    print(f"Traces → {traces_path}")

    with open(traces_path, "w") as traces_f:
        for sample in tqdm(samples):
            image = sample.frontal_image
            trace = {
                "uid": sample.uid,
                "frontal_image": image,
                "gt_normal": sample.is_normal,
                "gt_problems": sample.problems,
            }

            # Level 1
            pred_normal, raw_l1 = evaluator.predict_normal(image)
            trace["l1_raw"] = raw_l1
            trace["l1_pred"] = pred_normal
            if pred_normal is None:
                skipped_l1 += 1
            else:
                l1_true.append(sample.is_normal)
                l1_pred.append(pred_normal)

            # Level 2
            pred_labels, raw_l2 = evaluator.predict_labels(image, EVAL_LABELS)
            trace["l2_raw"] = raw_l2
            trace["l2_pred"] = {k: v for k, v in pred_labels.items()}
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

            # Level 3 — free-text report
            report, raw_l3 = evaluator.predict_report(image)
            trace["l3_raw"] = raw_l3
            trace["l3_findings"] = report["findings"]
            trace["l3_impression"] = report["impression"]

            traces_f.write(json.dumps(trace) + "\n")
            traces_f.flush()

    print(f"\nLevel 1 skipped: {skipped_l1}  |  Level 2 skipped: {skipped_l2}")

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

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics  → {out_path}")
    print(f"Traces   → {traces_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(device=args.device, output=args.output)


if __name__ == "__main__":
    main()
