"""
L2 eval callback: after each eval pass, run tag_classification inference on a
fixed subset of the validation set and log per-label and macro F1 to ClearML.

Only executes on local_rank 0 to avoid duplicate work in DDP.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from transformers import TrainerCallback

from evals.vocab import EVAL_LABELS
from training.tasks import PROMPTS


def _parse_labels(raw: str, labels: List[str]) -> dict:
    result = {l: None for l in labels}
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for label in labels:
                if label in data:
                    result[label] = bool(data[label])
                else:
                    for k, v in data.items():
                        if k.lower() == label.lower():
                            result[label] = bool(v)
                            break
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("r") or item.get("finding") or item.get("label") or item.get("name")
                val  = item.get("f") if item.get("f") is not None else item.get("present")
                if name and val is not None:
                    for label in labels:
                        if str(name).lower() == label.lower():
                            result[label] = bool(val)
    except Exception:
        pass
    return result


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


class L2EvalCallback(TrainerCallback):
    """Runs tag_classification inference on a fixed val subset after each eval."""

    def __init__(
        self,
        val_samples: list,
        processor,
        device: str,
        clearml_logger,
        n_samples: int = 150,
        batch_size: int = 8,
        image_size: tuple = (512, 512),
        seed: int = 42,
    ):
        self.processor = processor
        self.device = device
        self.logger = clearml_logger
        self.batch_size = batch_size
        self.image_size = image_size
        self.prompt = PROMPTS["tag_classification"]

        # Fixed balanced subset: equal normal/abnormal
        rng = random.Random(seed)
        normal   = [s for s in val_samples if not s["problems"]]
        abnormal = [s for s in val_samples if s["problems"]]
        n_each   = n_samples // 2
        self.subset = (rng.sample(normal,   min(n_each, len(normal)))
                     + rng.sample(abnormal, min(n_each, len(abnormal))))
        print(f"L2EvalCallback: {len(self.subset)} val samples "
              f"({sum(1 for s in self.subset if not s['problems'])} normal, "
              f"{sum(1 for s in self.subset if s['problems'])} abnormal)")

    def _load_image(self, path: Optional[str]) -> Optional[Image.Image]:
        if not path or not Path(path).exists():
            return None
        return Image.open(path).convert("RGB").resize(self.image_size, Image.BILINEAR)

    def _run_inference(self, model) -> List[dict]:
        """Run tag_classification on subset and return [{uid, pred, gt}, ...]."""
        results = []

        for i in range(0, len(self.subset), self.batch_size):
            batch = self.subset[i:i + self.batch_size]
            texts, images, valid = [], [], []

            for s in batch:
                img = self._load_image(s.get("frontal"))
                if img is None:
                    continue
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": self.prompt},
                ]}]
                texts.append(self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                ))
                images.append(img)
                valid.append(s)

            if not texts:
                continue

            inputs = self.processor(
                text=texts, images=images, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=384, do_sample=False
                )

            input_len = inputs["input_ids"].shape[1]
            for s, out in zip(valid, output_ids):
                raw = self.processor.decode(out[input_len:], skip_special_tokens=True).strip()
                pred = _parse_labels(raw, EVAL_LABELS)
                results.append({
                    "uid":      s.get("uid"),
                    "pred":     pred,
                    "problems": s.get("problems", []),
                })

        return results

    def _compute_metrics(self, results: List[dict]) -> dict:
        counters = {l: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for l in EVAL_LABELS}
        skipped = 0

        for r in results:
            preds    = r["pred"]
            problems = set(r["problems"])
            if all(v is None for v in preds.values()):
                skipped += 1
                continue
            for label in EVAL_LABELS:
                gt   = label in problems
                pred = preds.get(label)
                if pred is None:
                    continue
                c = counters[label]
                if gt and pred:     c["tp"] += 1
                elif gt and not pred:   c["fn"] += 1
                elif not gt and pred:   c["fp"] += 1
                else:                   c["tn"] += 1

        per_label = {}
        f1_scores = []
        for label in EVAL_LABELS:
            c  = counters[label]
            f1 = _f1(c["tp"], c["fp"], c["fn"])
            per_label[label] = {"f1": f1, **c}
            f1_scores.append(f1)

        return {
            "macro_f1":          sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            "macro_sensitivity": sum(c["tp"] / (c["tp"] + c["fn"])
                                     for c in counters.values()
                                     if (c["tp"] + c["fn"]) > 0)
                                 / max(1, sum(1 for c in counters.values() if (c["tp"]+c["fn"])>0)),
            "per_label":         per_label,
            "skipped":           skipped,
        }

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if args.local_rank != 0 or model is None or self.logger is None:
            return

        # Unwrap DDP / PEFT wrappers if needed
        unwrapped = model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        was_training = unwrapped.training
        unwrapped.eval()
        try:
            results = self._run_inference(unwrapped)
            metrics = self._compute_metrics(results)
            step    = state.global_step

            self.logger.report_scalar("eval/l2_macro_f1",         "macro_f1",    metrics["macro_f1"],          step)
            self.logger.report_scalar("eval/l2_macro_sensitivity", "sensitivity", metrics["macro_sensitivity"],  step)

            for label, m in metrics["per_label"].items():
                self.logger.report_scalar("eval/l2_per_label_f1", label, m["f1"], step)

            print(f"  L2 eval (step {step}): macro_f1={metrics['macro_f1']:.3f}  "
                  f"sensitivity={metrics['macro_sensitivity']:.3f}  "
                  f"skipped={metrics['skipped']}/{len(self.subset)}")
        finally:
            if was_training:
                unwrapped.train()
