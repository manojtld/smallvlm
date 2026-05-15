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
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from .dataset import load_test_split
from .inference import QwenEvaluator
from .metrics import (
    binary_metrics, multilabel_metrics,
    print_binary_results, print_multilabel_results,
)
from .vocab import EVAL_LABELS

RESULTS_DIR = Path(__file__).parent / "results"


def _worker(rank: int, samples: list, traces_path: str, lock: mp.Lock, model_id: str):
    """Run inference on a shard of samples on GPU `rank`, streaming traces to shared file."""
    device = f"cuda:{rank}"
    evaluator = QwenEvaluator(device=device, model_id=model_id)

    for sample in tqdm(samples, desc=f"GPU{rank}", position=rank, leave=False):
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

        with lock:
            with open(traces_path, "a") as f:
                f.write(json.dumps(trace) + "\n")


def run(n_gpus: int = 1, output: str = None, model_id: str = "Qwen/Qwen3.5-0.8B") -> dict:
    samples = load_test_split()
    out_path = Path(output) if output else RESULTS_DIR / "baseline_qwen35_08b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path = out_path.with_suffix(".traces.jsonl")

    # Clear any stale traces from a previous run
    traces_path.write_text("")

    shards = [samples[i::n_gpus] for i in range(n_gpus)]
    print(f"\nRunning eval on {len(samples)} samples across {n_gpus} GPU(s)")
    print(f"Shard sizes: {[len(s) for s in shards]}")
    print(f"Traces (live) → {traces_path}\n")

    mp.set_start_method("spawn", force=True)
    lock = mp.Lock()

    if n_gpus == 1:
        _worker(0, shards[0], str(traces_path), lock, model_id)
    else:
        processes = []
        for rank in range(n_gpus):
            p = mp.Process(target=_worker, args=(rank, shards[rank], str(traces_path), lock, model_id))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    all_traces = [json.loads(l) for l in traces_path.read_text().splitlines() if l.strip()]
    all_traces.sort(key=lambda t: t["uid"])
    traces_path.write_text("\n".join(json.dumps(t) for t in all_traces) + "\n")
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
        "model": model_id,
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
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    args = parser.parse_args()
    run(n_gpus=args.gpus, output=args.output, model_id=args.model)


if __name__ == "__main__":
    main()
