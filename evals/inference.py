"""
Run Qwen3.5-0.8B inference on CXR images for eval.

Level 1 prompt: normal/abnormal classification
Level 2 prompt: closed-vocab finding presence/absence
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-0.8B"

LEVEL1_PROMPT = (
    "You are a radiologist. Look at this chest X-ray and answer with a single word only.\n"
    "Is this chest X-ray normal or abnormal?\n"
    "Answer: Normal or Abnormal"
)

LEVEL2_PROMPT_TEMPLATE = (
    "You are a radiologist. Look at this chest X-ray.\n"
    "For each finding below, state whether it is present (true) or absent (false).\n"
    "Output ONLY a valid JSON object — no prose, no markdown fences.\n\n"
    "Findings:\n{labels}\n\n"
    "Output format: {{\"Finding Name\": true/false, ...}}"
)


class QwenEvaluator:
    def __init__(self, device: str = "cuda:0", dtype=torch.bfloat16):
        print(f"Loading {MODEL_ID} on {device}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        print("Model loaded.")

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def predict_normal(self, image_path: str) -> Optional[bool]:
        """Returns True if model predicts normal, False if abnormal, None if unparseable."""
        if not image_path or not Path(image_path).exists():
            return None
        image = self._load_image(image_path)
        response = self._generate(image, LEVEL1_PROMPT, max_new_tokens=10)
        lower = response.lower()
        if "normal" in lower and "abnormal" not in lower:
            return True
        if "abnormal" in lower:
            return False
        return None

    def predict_labels(self, image_path: str, labels: List[str]) -> Dict[str, Optional[bool]]:
        """Returns {label: True/False/None} for each label in vocab."""
        result = {label: None for label in labels}
        if not image_path or not Path(image_path).exists():
            return result

        image = self._load_image(image_path)
        label_list = "\n".join(f"- {l}" for l in labels)
        prompt = LEVEL2_PROMPT_TEMPLATE.format(labels=label_list)
        response = self._generate(image, prompt, max_new_tokens=256)

        # Strip markdown fences if present
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(response)
            # Model sometimes returns a list of single-key dicts — flatten it
            if isinstance(data, list):
                merged = {}
                for item in data:
                    if isinstance(item, dict):
                        merged.update(item)
                data = merged
            if isinstance(data, dict):
                for label in labels:
                    if label in data:
                        result[label] = bool(data[label])
                    else:
                        for k, v in data.items():
                            if k.lower() == label.lower():
                                result[label] = bool(v)
                                break
        except (json.JSONDecodeError, TypeError):
            pass

        return result
