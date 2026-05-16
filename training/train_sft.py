"""
SFT training script for Qwen3.5-2B on CXR reports.

Uses HuggingFace Trainer (not TRL SFTTrainer) with a custom data collator
to handle variable numbers of images per sample and curriculum task sampling.

Usage:
  accelerate launch --config_file training/configs/accelerate.yaml \\
      training/train_sft.py --phase 1 \\
      --output /raid3/manoj/smallvlm/checkpoints/sft_phase1_2b

  # Continue from previous phase checkpoint:
  accelerate launch --config_file training/configs/accelerate.yaml \\
      training/train_sft.py --phase 2 \\
      --base /raid3/manoj/smallvlm/checkpoints/sft_phase1_2b \\
      --output /raid3/manoj/smallvlm/checkpoints/sft_phase2_2b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.collator import CXRCollator
from training.dataset import CXRSFTDataset
from training.l2_eval import L2EvalCallback
from training.metrics import make_clearml_callback

# ── LoRA targets: both DeltaNet linear_attn and standard self_attn layers ─────
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # standard self_attn
    "in_proj_qkv", "out_proj",                      # DeltaNet linear_attn
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase",       type=int,   required=True, choices=[1, 2, 3, 4, 5, 6])
    CKPT_ROOT = os.environ.get("SMALLVLM_DATA", "/raid3/manoj/smallvlm") + "/models/checkpoints"
    p.add_argument("--base",        default=CKPT_ROOT + "/base_qwen35_2b")
    p.add_argument("--output",      required=True)
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--batch-size",  type=int,   default=2, help="Per-device batch size")
    p.add_argument("--grad-accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lora-r",      type=int,   default=32)
    p.add_argument("--lora-alpha",  type=int,   default=32)
    p.add_argument("--max-len",     type=int,   default=8192)
    p.add_argument("--eval-steps",  type=int,   default=None,
                   help="Eval every N steps. Default: every 0.5 epochs.")
    p.add_argument("--clf-loss-weight", type=float, default=1.0,
                   help="Loss multiplier for tag_classification task (default 1.0).")
    p.add_argument("--clearml-project", default="smallvlm")
    p.add_argument("--clearml-task",    default=None)
    return p.parse_args()


def main():
    args = parse_args()
    task_name = args.clearml_task or f"sft_phase{args.phase}_2b"

    clearml_cb = make_clearml_callback(
        project=args.clearml_project,
        task_name=task_name,
        phase=args.phase,
    )

    print(f"\n=== SFT Phase {args.phase} ===")
    print(f"Base      : {args.base}")
    print(f"Output    : {args.output}")

    # ── Model + processor ─────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Datasets + collator ───────────────────────────────────────────────────
    train_ds = CXRSFTDataset(phase=args.phase, split="train", augment=True)
    val_ds   = CXRSFTDataset(phase=args.phase, split="val",   augment=False)
    collator = CXRCollator(processor, max_length=args.max_len)

    # Eval every 0.5 epochs by default
    steps_per_epoch = len(train_ds) // (args.batch_size * 4 * args.grad_accum)  # 4 GPUs
    eval_steps = args.eval_steps if args.eval_steps else max(1, steps_per_epoch // 2)
    print(f"steps/epoch={steps_per_epoch}  eval every {eval_steps} steps (0.5 epochs)")

    # Weighted sampler: 40% normal, 60% abnormal weighted by inverse label freq
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    print(f"WeightedRandomSampler: {sum(1 for w in sample_weights if w > 0)} non-zero weights")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )

    # Group task types into families for cleaner ClearML plots
    TASK_FAMILY = {
        "primitive_observations": "primitives",
        "tag_classification":     "classification",
        "mesh_tags":              "mesh_tags",
        "findings":               "findings_impression",
        "impression":             "findings_impression",
        "structured_json":        "json",
    }

    # Per-task loss weights — upweight classification since it's sampled less often
    TASK_LOSS_WEIGHT = {
        "findings":          1.0,
        "impression":        1.0,
        "tag_classification": float(args.clf_loss_weight),
        "primitive_observations": 1.0,
        "mesh_tags":         1.0,
        "structured_json":   1.0,
    }

    # Grab the ClearML logger from the callback so compute_loss can use it
    _clearml_logger = getattr(clearml_cb, "_logger", None)
    # Try to extract it after Task.init via the callback internals
    try:
        from clearml import Task
        _task = Task.current_task()
        if _task:
            _clearml_logger = _task.get_logger()
    except Exception:
        pass

    class WeightedTrainer(Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return sampler

        # Accumulators for per-task losses — flushed when HF Trainer logs
        _task_loss_accum: dict = {}     # train: {family: [losses]}
        _eval_task_loss_accum: dict = {}  # eval: {family: [losses]}

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            task_types = inputs.pop("task_types", None)
            outputs = model(**inputs)

            # Sample-level mean loss: every sample contributes equally regardless of
            # target length. Computed with the same label shift the model uses internally
            # (logit[i] predicts label[i+1]).
            logits = outputs.logits                              # (B, T, V)
            labels = inputs["labels"]                            # (B, T)
            shift_logits = logits[:, :-1, :].contiguous()       # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()            # (B, T-1)
            per_tok = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)                            # (B, T-1)
            valid_mask = (shift_labels != -100).float()
            per_sample = (per_tok * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
            # Apply per-task loss weights before averaging
            if task_types:
                weights = torch.tensor(
                    [TASK_LOSS_WEIGHT.get(t, 1.0) for t in task_types],
                    device=per_sample.device, dtype=per_sample.dtype
                )
                loss = (per_sample * weights).mean()
            else:
                loss = per_sample.mean()

            # Accumulate per-task losses for ClearML
            if task_types:
                try:
                    is_eval = return_outputs
                    accum = self._eval_task_loss_accum if is_eval else self._task_loss_accum
                    for i, task in enumerate(task_types):
                        fam = TASK_FAMILY.get(task, task)
                        accum.setdefault(fam, []).append(per_sample[i].item())
                except Exception:
                    pass

            return (loss, outputs) if return_outputs else loss

        def _flush_task_losses(self, accum: dict, split: str, step: int):
            if not accum or not _clearml_logger:
                return
            try:
                for fam, vals in accum.items():
                    _clearml_logger.report_scalar(
                        title=f"{split}/loss_by_task",
                        series=fam,
                        value=sum(vals) / len(vals),
                        iteration=step,
                    )
            except Exception:
                pass
            accum.clear()

        def log(self, logs, *args, **kwargs):
            # Trainer.log() is called every logging_steps — flush train per-task losses
            super().log(logs, *args, **kwargs)
            self._flush_task_losses(self._task_loss_accum, "train", self.state.global_step)

        def evaluate(self, *args, **kwargs):
            # Trainer.evaluate() runs the full eval loop — flush eval per-task losses after
            result = super().evaluate(*args, **kwargs)
            self._flush_task_losses(self._eval_task_loss_accum, "eval", self.state.global_step)
            return result

    l2_cb = L2EvalCallback(
        val_samples=val_ds.samples,
        processor=processor,
        device=training_args.device.type + ":0",
        clearml_logger=_clearml_logger,
        n_samples=150,
        batch_size=8,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[clearml_cb, l2_cb],
    )

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
