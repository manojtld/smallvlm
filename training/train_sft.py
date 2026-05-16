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
from training.metrics import make_clearml_callback

# ── LoRA targets: both DeltaNet linear_attn and standard self_attn layers ─────
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # standard self_attn
    "in_proj_qkv", "out_proj",                      # DeltaNet linear_attn
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase",       type=int,   required=True, choices=[1, 2, 3, 4, 5])
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

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            task_types = inputs.pop("task_types", None)
            outputs = model(**inputs)
            loss = outputs.loss

            # Per-task loss logging — only on rank 0 to avoid duplicate logs
            if task_types and _clearml_logger and self.state.global_step % 10 == 0:
                try:
                    labels = inputs["labels"]                       # (B, T)
                    logits = outputs.logits                          # (B, T, V)
                    per_tok = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(labels.shape)                            # (B, T)
                    valid_mask = (labels != -100).float()
                    per_sample = (per_tok * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)

                    family_losses: dict = {}
                    for i, task in enumerate(task_types):
                        fam = TASK_FAMILY.get(task, task)
                        family_losses.setdefault(fam, []).append(per_sample[i].item())

                    step = self.state.global_step
                    for fam, vals in family_losses.items():
                        _clearml_logger.report_scalar(
                            title=f"train/loss_by_task",
                            series=fam,
                            value=sum(vals) / len(vals),
                            iteration=step,
                        )
                except Exception:
                    pass  # never crash training over logging

            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[clearml_cb],
    )

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
