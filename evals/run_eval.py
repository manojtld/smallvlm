"""
Run the CXR evaluation benchmark on a model.

Usage:
  python -m evals.run_eval
  python -m evals.run_eval --gpus 8 --output evals/results/baseline.json

Data-parallel across multiple GPUs: each GPU gets a model replica and processes
an equal shard of the test set. Results are merged and metrics computed jointly.

Outputs (always written):
  <output>               — aggregated metrics JSON
  <output>.traces.jsonl  — per-sample traces (uid, image, gt, raw responses, predictions)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from .dataset import EvalSample, load_test_split
from .inference import QwenEvaluator
from .metrics import (
    binary_metrics, multilabel_metrics,
    print_binary_results, print_multilabel_results,
)
from .vocab import EVAL_LABELS

RESULTS_DIR = Path(__file__).parent / "results"


def _worker(rank: int, samples: list, tmp_path: str):
    """Run inference on a shard of samples on GPU `rank`, write traces to tmp_path."""
    device = f"cuda:{rank}"
    evaluator = QwenEvaluator(device=device)

    with open(tmp_path, "w") as f:
        for sample in tqdm(samples, desc=f"GPU {rank}", position=rank):
            image = sample.frontal_image
            trace = {
                "uid": sample.uid,
                "frontal_image": image,
                "gt_normal": sample.is_normal,
                "gt_problems": sample.problems,
            }

            pred_normal, raw_l1 = evaluator.predict_normal(image)
            trace["l1_raw"] = raw_l1
            trace["l1_pred"] = pred_normal

            pred_labels, raw_l2 = evaluator.predict_labels(image, EVAL_LABELS)
            trace["l2_raw"] = raw_l2
            trace["l2_pred"] = {k: v for k, v in pred_labels.items()}

            report, raw_l3 = evaluator.predict_report(image)
            trace["l3_raw"] = raw_l3
            trace["l3_findings"] = report["findings"]
            trace["l3_impression"] = report["impression"]

            f.write(json.dumps(trace) + "\n")
            f.flush()


def run(n_gpus: int = 1, output: str = None) -> dict:
    samples = load_test_split()
    out_path = Path(output) if output else RESULTS_DIR / "baseline_qwen35_08b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path = out_path.with_suffix(".traces.jsonl")

    # Split samples evenly across GPUs
    shards = [samples[i::n_gpus] for i in range(n_gpus)]
    tmp_files = [tempfile.mktemp(suffix=".jsonl") for _ in range(n_gpus)]

    print(f"\nRunning eval on {len(samples)} samples across {n_gpus} GPU(s)")
    print(f"Shard sizes: {[len(s) for s in shards]}")
    print(f"Traces → {traces_path}\n")

    if n_gpus == 1:
        _worker(0, shards[0], tmp_files[0])
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(n_gpus):
            p = mp.Process(target=_worker, args=(rank, shards[rank], tmp_files[rank]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # Merge shard traces into final traces file
    all_traces = []
    for tmp in tmp_files:
        if Path(tmp).exists():
            with open(tmp) as f:
                all_traces.extend(json.loads(l) for l in f if l.strip())
            Path(tmp).unlink()

    all_traces.sort(key=lambda t: t["uid"])
    with open(traces_path, "w") as f:
        for t in all_traces:
            f.write(json.dumps(t) + "\n")
    print(f"\nWrote {len(all_traces)} traces → {traces_path}")

    # Compute metrics
    l1_true, l1_pred = [], []
    l2_true, l2_pred = defaultdict(list), defaultdict(list)
    skipped_l1 = skipped_l2 = 0

    for t in all_traces:
        pred = t.get("l1_pred")
        if pred is None:
            skipped_l1 += 1
        else:
            l1_true.append(t["gt_normal"])
            l1_pred.append(pred)

        preds = t.get("l2_pred", {})
        has_any = any(v is not None for v in preds.values())
        if not has_any:
            skipped_l2 += 1
        else:
            for label in EVAL_LABELS:
                gt = label in t["gt_problems"]
                pred_l = preds.get(label)
                if pred_l is not None:
                    l2_true[label].append(gt)
                    l2_pred[label].append(pred_l)

    print(f"Level 1 skipped: {skipped_l1}  |  Level 2 skipped: {skipped_l2}")

    l1_metrics = binary_metrics(l1_true, l1_pred)
    l2_results = multilabel_metrics(dict(l2_true), dict(l2_pred), EVAL_LABELS)

    print_binary_results("Level 1 — Normal/Abnormal Classification", l1_metrics)
    print_multilabel_results(l2_results, EVAL_LABELS)

    results = {
        "model": "Qwen/Qwen3.5-0.8B",
        "n_samples": len(samples),
        "n_gpus": n_gpus,
        "level1": l1_metrics,
        "level2": l2_results,
    }

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics  → {out_path}")
    print(f"Traces   → {traces_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(n_gpus=args.gpus, output=args.output)


if __name__ == "__main__":
    main()
