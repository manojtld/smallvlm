"""
ClearML callback and eval metric computation for SFT training.

Tracked every step (via callback):
  train/loss, train/grad_norm, train/lr, train/throughput_tokens_per_sec

Tracked every eval_steps (via compute_eval_metrics, sampled on val set):
  eval/loss, eval/json_parse_rate, eval/rouge_l_primitives, eval/rouge_l_findings

Tracked every epoch (via callback, runs L1/L2 eval on 100-sample subset):
  eval/normal_abnormal_f1, eval/finding_macro_f1
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def _rouge_l(pred: str, ref: str) -> float:
    """Simple token-level ROUGE-L (LCS / max_len)."""
    pred_toks = pred.lower().split()
    ref_toks  = ref.lower().split()
    if not pred_toks or not ref_toks:
        return 0.0
    # LCS via DP
    m, n = len(ref_toks), len(pred_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref_toks[i-1] == pred_toks[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall    = lcs / m if m else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_eval_metrics(predictions: list[dict]) -> dict:
    """
    Compute eval metrics from a list of {task, target, prediction} dicts.
    Called by the trainer's compute_metrics or manually in the callback.
    """
    json_total, json_ok = 0, 0
    rouge_prim, rouge_find = [], []

    for p in predictions:
        task   = p.get("task", "")
        target = p.get("target", "")
        pred   = p.get("prediction", "")

        if task in ("tag_classification", "structured_json", "primitive_observations", "mesh_tags"):
            json_total += 1
            try:
                json.loads(pred)
                json_ok += 1
            except Exception:
                pass

        if task == "primitive_observations":
            rouge_prim.append(_rouge_l(pred, target))

        if task == "findings":
            rouge_find.append(_rouge_l(pred, target))

    metrics = {}
    if json_total:
        metrics["eval/json_parse_rate"] = json_ok / json_total
    if rouge_prim:
        metrics["eval/rouge_l_primitives"] = sum(rouge_prim) / len(rouge_prim)
    if rouge_find:
        metrics["eval/rouge_l_findings"] = sum(rouge_find) / len(rouge_find)

    return metrics


def make_clearml_callback(project: str, task_name: str, phase: int):
    """Create and return a ClearML logging callback."""
    try:
        from clearml import Task
        task = Task.init(project_name=project, task_name=task_name, reuse_last_task_id=False)
        task.connect({"phase": phase})
        logger = task.get_logger()
    except Exception as e:
        print(f"ClearML init failed ({e}), logging disabled.")
        task = None
        logger = None

    class ClearMLCallback(TrainerCallback):
        def __init__(self):
            self._step_start = None

        def on_step_begin(self, args, state, control, **kwargs):
            self._step_start = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logger is None or logs is None:
                return
            step = state.global_step
            for k, v in logs.items():
                if not isinstance(v, (int, float)):
                    continue
                # Map HF Trainer log keys to our naming convention
                series = k.replace("train_", "train/").replace("eval_", "eval/")
                logger.report_scalar(title=series, series=series, value=float(v), iteration=step)

            # Throughput
            if self._step_start and "loss" in logs:
                elapsed = time.time() - self._step_start
                # approximate tokens per second
                tps = (args.per_device_train_batch_size * args.gradient_accumulation_steps
                       * args.max_steps * 512) / max(elapsed, 1e-6)
                logger.report_scalar("train/throughput_tokens_per_sec", "train/throughput_tokens_per_sec",
                                     tps, step)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if logger is None or metrics is None:
                return
            step = state.global_step
            for k, v in metrics.items():
                if not isinstance(v, (int, float)):
                    continue
                series = k if k.startswith("eval/") else f"eval/{k}"
                logger.report_scalar(title=series, series=series, value=float(v), iteration=step)

        def on_train_end(self, args, state, control, **kwargs):
            if task is not None:
                task.close()

    return ClearMLCallback()
