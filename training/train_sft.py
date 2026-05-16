"""
SFT training script for Qwen3.5-2B on CXR reports.

Usage:
  # Single phase
  accelerate launch --config_file training/configs/accelerate.yaml \\
      training/train_sft.py --phase 1 --output /raid/manoj/smallvlm/checkpoints/sft_phase1

  # Continue from previous phase
  accelerate launch --config_file training/configs/accelerate.yaml \\
      training/train_sft.py --phase 2 --base /raid/manoj/smallvlm/checkpoints/sft_phase1 \\
      --output /raid/manoj/smallvlm/checkpoints/sft_phase2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import CXRSFTDataset
from training.metrics import compute_eval_metrics, make_clearml_callback

# ── LoRA target modules ────────────────────────────────────────────────────────
# Both DeltaNet linear_attn layers and standard self_attn layers
LORA_TARGET_MODULES = [
    # Standard attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # DeltaNet linear attention
    "in_proj_qkv", "out_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--base", default="/raid/manoj/smallvlm/checkpoints/base_qwen35_2b",
                   help="Base model or previous phase checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--eval-steps", type=int, default=200, help="Run eval every N steps")
    p.add_argument("--clearml-project", default="smallvlm")
    p.add_argument("--clearml-task", default=None)
    return p.parse_args()


def load_model_and_processor(base_path: str, lora_r: int, lora_alpha: int):
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def collate_fn(batch, processor, max_seq_len):
    """Convert list of dataset samples to model inputs."""
    valid = [b for b in batch if b is not None]
    if not valid:
        return None

    texts, images_batch = [], []
    for sample in valid:
        messages = sample["messages"]
        # Extract images from messages
        imgs = []
        for msg in messages:
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        imgs.append(part["image"])
        images_batch.append(imgs)
        # Format chat template (processor handles image token injection)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Flatten image list
    all_images = [img for imgs in images_batch for img in imgs]

    inputs = processor(
        text=texts,
        images=all_images if all_images else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )

    # For SFT, labels = input_ids with prompt tokens masked to -100
    inputs["labels"] = inputs["input_ids"].clone()

    return inputs


def main():
    args = parse_args()
    task_name = args.clearml_task or f"sft_phase{args.phase}_2b"

    # ClearML init — must happen before any model loading on rank 0
    clearml_cb = make_clearml_callback(
        project=args.clearml_project,
        task_name=task_name,
        phase=args.phase,
    )

    print(f"\n=== SFT Phase {args.phase} ===")
    print(f"Base model : {args.base}")
    print(f"Output     : {args.output}")

    model, processor = load_model_and_processor(args.base, args.lora_r, args.lora_alpha)

    train_ds = CXRSFTDataset(phase=args.phase, split="train", augment=True)
    val_ds   = CXRSFTDataset(phase=args.phase, split="val",   augment=False)

    sft_config = SFTConfig(
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
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # we handle logging via clearml_cb
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        callbacks=[clearml_cb],
    )

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
