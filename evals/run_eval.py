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

# Batch sizes per task — tune based on available VRAM.
# 0.8B model is ~1.7GB, leaving ~78GB free on H100 80GB.
L1_BATCH = 96   # 16 output tokens
L2_BATCH = 64   # 384 output tokens
L3_BATCH = 48   # 800 output tokens


def _batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _worker(rank: int, samples: list, traces_path: str, lock: mp.Lock, model_id: str):
    """
    Run all 3 tasks in batches (task-first order) on GPU `rank`.
    Writes a trace per sample to the shared traces file as each task completes.
    """
    from .inference import LEVEL2_PROMPT_TEMPLATE, LEVEL3_PROMPT, LEVEL1_PROMPT

    device = f"cuda:{rank}"
    evaluator = QwenEvaluator(device=device, model_id=model_id)

    image_paths = [s.frontal_image for s in samples]
    uids = [s.uid for s in samples]
    traces = {s.uid: {
        "uid": s.uid, "frontal_image": s.frontal_image,
        "gt_normal": s.is_normal, "gt_problems": s.problems,
        "l1_raw": "", "l1_pred": None,
        "l2_raw": "", "l2_pred": {k: None for k in EVAL_LABELS},
        "l3_raw": "", "l3_findings": "", "l3_impression": "",
    } for s in samples}

    l2_prompt = LEVEL2_PROMPT_TEMPLATE.format(labels="\n".join(f"- {l}" for l in EVAL_LABELS))

    # ── L1: normal/abnormal ──────────────────────────────────────────────────
    all_l1_raw = [""] * len(samples)
    for i, batch_paths in enumerate(tqdm(list(_batches(image_paths, L1_BATCH)),
                                         desc=f"GPU{rank} L1", position=rank, leave=False)):
        start = i * L1_BATCH
        raws = evaluator.generate_batch(batch_paths, LEVEL1_PROMPT, max_new_tokens=16)
        all_l1_raw[start:start + len(raws)] = raws

    for uid, raw in zip(uids, all_l1_raw):
        traces[uid]["l1_raw"] = raw
        traces[uid]["l1_pred"] = evaluator._parse_normal(raw)

    # ── L2: finding presence ─────────────────────────────────────────────────
    all_l2_raw = [""] * len(samples)
    for i, batch_paths in enumerate(tqdm(list(_batches(image_paths, L2_BATCH)),
                                         desc=f"GPU{rank} L2", position=rank, leave=False)):
        start = i * L2_BATCH
        raws = evaluator.generate_batch(batch_paths, l2_prompt, max_new_tokens=384)
        all_l2_raw[start:start + len(raws)] = raws

    for uid, raw in zip(uids, all_l2_raw):
        traces[uid]["l2_raw"] = raw
        traces[uid]["l2_pred"] = evaluator._parse_labels(raw, EVAL_LABELS)

    # ── L3: free-text report ─────────────────────────────────────────────────
    all_l3_raw = [""] * len(samples)
    for i, batch_paths in enumerate(tqdm(list(_batches(image_paths, L3_BATCH)),
                                         desc=f"GPU{rank} L3", position=rank, leave=False)):
        start = i * L3_BATCH
        raws = evaluator.generate_batch(batch_paths, LEVEL3_PROMPT,
                                         max_new_tokens=800, repetition_penalty=1.3)
        all_l3_raw[start:start + len(raws)] = raws

    for uid, raw in zip(uids, all_l3_raw):
        parsed = evaluator._parse_report(raw)
        traces[uid]["l3_raw"] = raw
        traces[uid]["l3_findings"] = parsed["findings"]
        traces[uid]["l3_impression"] = parsed["impression"]

    # ── Write all traces for this shard ──────────────────────────────────────
    with lock:
        with open(traces_path, "a") as f:
            for uid in uids:
                f.write(json.dumps(traces[uid]) + "\n")


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
