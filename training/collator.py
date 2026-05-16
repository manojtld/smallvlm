"""
Data collator for CXR SFT training.

Handles variable numbers of images per sample (1 or 2 views), applies the
processor to produce tokenized inputs, and masks prompt tokens in labels.
"""

from __future__ import annotations

from typing import List

import torch


class CXRCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[dict]) -> dict:
        valid = [b for b in batch if b is not None]
        if not valid:
            return {}

        texts, all_images = [], []

        for sample in valid:
            messages = sample["messages"]

            # Collect images from this sample
            sample_images = []
            for msg in messages:
                if isinstance(msg["content"], list):
                    for part in msg["content"]:
                        if part.get("type") == "image":
                            sample_images.append(part["image"])
            all_images.extend(sample_images)

            # Apply chat template (processor inserts <image> tokens)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Build labels — mask everything up to and including the last assistant turn start.
        # Strategy: find the assistant response tokens and only supervise those.
        labels = inputs["input_ids"].clone()

        # Mask pad tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask the prompt (user turn) — keep only the assistant response tokens.
        # We locate the assistant turn by finding where the model's response begins.
        # For Qwen, the assistant response follows <|im_start|>assistant\n
        assistant_token = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if assistant_token:
            for i, ids in enumerate(inputs["input_ids"]):
                # Find all positions where the assistant token sequence starts
                last_assistant_start = -1
                for j in range(len(ids) - len(assistant_token)):
                    if ids[j:j+len(assistant_token)].tolist() == assistant_token:
                        last_assistant_start = j
                # Mask everything up to and including the assistant header
                if last_assistant_start >= 0:
                    labels[i, :last_assistant_start + len(assistant_token)] = -100

        inputs["labels"] = labels
        return inputs
