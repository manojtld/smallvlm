"""
Qwen VLM inference for CXR eval.

Uses batched inference across a single GPU — each GPU replica gets a shard
of the dataset and processes it in batches to saturate VRAM.

Images are resized to IMAGE_SIZE before batching to ensure uniform patch counts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-0.8B"
IMAGE_SIZE = (512, 512)  # resize all images to this before batching

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

LEVEL3_PROMPT = (
    "You are a radiologist. Write a structured radiology report for this chest X-ray.\n\n"
    "FINDINGS:\n"
    "List each radiological observation on a separate line.\n\n"
    "IMPRESSION:\n"
    "Summarize the key clinical conclusions in 1-3 sentences."
)


class QwenEvaluator:
    def __init__(self, device: str = "cuda:0", dtype=torch.bfloat16, model_id: str = MODEL_ID):
        print(f"Loading {model_id} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        print("Model loaded.")

    def _load_image(self, path: str) -> Optional[Image.Image]:
        if not path or not Path(path).exists():
            return None
        return Image.open(path).convert("RGB").resize(IMAGE_SIZE)

    # ── Core batched generation ───────────────────────────────────────────────

    def generate_batch(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """
        Run `prompt` on a batch of images. Returns one response string per path.
        Paths that don't exist return an empty string.
        """
        results = [""] * len(image_paths)
        images, valid_idx = [], []

        for i, path in enumerate(image_paths):
            img = self._load_image(path)
            if img is not None:
                images.append(img)
                valid_idx.append(i)

        if not images:
            return results

        texts = []
        for img in images:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt},
            ]}]
            texts.append(self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))

        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
            )

        for i, (orig_idx, out) in enumerate(zip(valid_idx, output_ids)):
            new_tokens = out[input_len:]
            results[orig_idx] = self.processor.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

        return results

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _parse_normal(self, raw: str) -> Optional[bool]:
        lower = raw.lower()
        if "normal" in lower and "abnormal" not in lower:
            return True
        if "abnormal" in lower:
            return False
        return None

    def _parse_labels(self, raw: str, labels: List[str]) -> Dict[str, Optional[bool]]:
        result = {label: None for label in labels}
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
                    val  = item.get("f") or item.get("present") or item.get("value") or item.get("found")
                    if name is None or val is None:
                        if len(item) == 1:
                            name, val = next(iter(item.items()))
                        else:
                            continue
                    for label in labels:
                        if str(name).lower() == label.lower():
                            result[label] = bool(val)
                            break
        except (json.JSONDecodeError, TypeError):
            pass
        return result

    def _parse_report(self, raw: str) -> Dict[str, str]:
        section_re = re.compile(r"^\**\s*(FINDINGS|IMPRESSION)\s*:?\**\s*:?\s*", re.IGNORECASE)
        findings_lines, impression_lines = [], []
        current = None
        for line in raw.splitlines():
            stripped = line.strip()
            m = section_re.match(stripped)
            if m:
                current = m.group(1).lower()
                remainder = stripped[m.end():].strip().lstrip("*- ")
                if remainder:
                    (findings_lines if current == "findings" else impression_lines).append(remainder)
                continue
            content = stripped.lstrip("*- ")
            if not content:
                continue
            if current == "findings":
                findings_lines.append(content)
            elif current == "impression":
                impression_lines.append(content)
        return {
            "l3_findings": " ".join(findings_lines),
            "l3_impression": " ".join(impression_lines),
        }

    # ── Single-sample API (kept for interactive use) ──────────────────────────

    def predict_normal(self, image_path: str) -> Tuple[Optional[bool], str]:
        raws = self.generate_batch([image_path], LEVEL1_PROMPT, max_new_tokens=16)
        raw = raws[0]
        return self._parse_normal(raw), raw

    def predict_labels(self, image_path: str, labels: List[str]) -> Tuple[Dict[str, Optional[bool]], str]:
        raws = self.generate_batch([image_path], LEVEL2_PROMPT_TEMPLATE.format(
            labels="\n".join(f"- {l}" for l in labels)), max_new_tokens=384)
        raw = raws[0]
        return self._parse_labels(raw, labels), raw

    def predict_report(self, image_path: str) -> Tuple[Dict[str, str], str]:
        raws = self.generate_batch([image_path], LEVEL3_PROMPT, max_new_tokens=800,
                                   repetition_penalty=1.3)
        raw = raws[0]
        parsed = self._parse_report(raw)
        return {"findings": parsed["l3_findings"], "impression": parsed["l3_impression"]}, raw
