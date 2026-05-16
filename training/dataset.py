"""
CXR SFT dataset.

Each sample produces a conversation dict with image(s) + prompt → target.
The task type is sampled uniformly from the active phase's task list.
Train/test split is enforced using the fixed split from evals/data/test_split.json.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset

from preprocessing.llm_formatter import load_canonical
from training.augmentations import prepare_views
from training.tasks import PHASES, PROMPTS, build_target

DATA_DIR = Path(os.environ.get("SMALLVLM_DATA", "/raid3/manoj/smallvlm"))
CANONICAL_PATH = DATA_DIR / "data" / "canonical.jsonl"
PROJECTIONS_CSV = DATA_DIR / "raddar/chest-xrays-indiana-university/versions/2/indiana_projections.csv"
REPORTS_CSV     = DATA_DIR / "raddar/chest-xrays-indiana-university/versions/2/indiana_reports.csv"
IMAGE_DIR       = DATA_DIR / "raddar/chest-xrays-indiana-university/versions/2/images/images_normalized"
TEST_SPLIT_PATH = Path("evals/data/test_split.json")


def _load_projections() -> dict:
    df = pd.read_csv(PROJECTIONS_CSV)
    mapping: dict = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        proj = str(row.get("projection", "")).lower()
        fname = str(row.get("filename", ""))
        if not fname:
            continue
        path = str(IMAGE_DIR / fname)
        if uid not in mapping:
            mapping[uid] = {}
        if "frontal" in proj or "pa" in proj:
            mapping[uid].setdefault("frontal", path)
        elif "lateral" in proj:
            mapping[uid].setdefault("lateral", path)
        else:
            mapping[uid].setdefault("frontal", path)
    return mapping


def _load_problems() -> dict:
    """Return {uid: [problem, ...]} from indiana_reports.csv Problems column."""
    df = pd.read_csv(REPORTS_CSV)
    result = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        raw = str(row.get("Problems", "") or "")
        problems = [t.strip() for t in raw.split(";") if t.strip() and t.strip().lower() != "normal"]
        result[uid] = problems
    return result


class CXRSFTDataset(Dataset):
    column_names = None  # TRL SFTTrainer checks for this; None → inferred from first item keys

    def __init__(
        self,
        phase: int,
        split: str = "train",
        augment: bool = True,
        drop_prob: float = 0.2,
        image_size: tuple = (512, 512),
        seed: int = 42,
    ):
        assert split in ("train", "val")
        assert phase in PHASES

        self.tasks = PHASES[phase]
        self.augment = augment and (split == "train")
        self.drop_prob = drop_prob if split == "train" else 0.0
        self.image_size = image_size
        self.rng = random.Random(seed)

        # Load test UIDs — everything else is training
        test_uids = set(json.loads(TEST_SPLIT_PATH.read_text()))

        # Load data
        reports = load_canonical(CANONICAL_PATH)
        projections = _load_projections()
        problems = _load_problems()

        self.samples = []
        for r in reports:
            in_test = r.uid in test_uids
            if split == "train" and in_test:
                continue
            if split == "val" and not in_test:
                continue
            proj = projections.get(r.uid, {})
            self.samples.append({
                "report": r,
                "frontal": proj.get("frontal"),
                "lateral": proj.get("lateral"),
                "problems": problems.get(r.uid, []),
            })

        print(f"CXRSFTDataset phase={phase} split={split}: {len(self.samples)} samples, tasks={self.tasks}")
        if split == "train":
            self._print_distribution()

    def _print_distribution(self):
        n_normal = sum(1 for s in self.samples if not s["problems"])
        print(f"  normal={n_normal} ({100*n_normal/len(self.samples):.1f}%)  "
              f"abnormal={len(self.samples)-n_normal} ({100*(len(self.samples)-n_normal)/len(self.samples):.1f}%)")

    def get_sample_weights(self) -> List[float]:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Strategy:
          - Normal samples: pooled weight = 0.4 (40% of draws go to normals)
          - Abnormal samples: pooled weight = 0.6, distributed by inverse label freq
            so rarer findings get proportionally more draws within the abnormal pool.
        """
        from evals.vocab import EVAL_LABELS
        from collections import Counter

        normal_idx, abnormal_idx = [], []
        for i, s in enumerate(self.samples):
            if not s["problems"]:
                normal_idx.append(i)
            else:
                abnormal_idx.append(i)

        # Per-label frequency for inverse-frequency weighting within abnormals
        label_counts: Counter = Counter()
        for i in abnormal_idx:
            for p in self.samples[i]["problems"]:
                if p in EVAL_LABELS:
                    label_counts[p] += 1

        def abnormal_weight(s) -> float:
            # Weight = sum of inverse frequencies of present eval labels
            # Falls back to 1.0 if no eval labels match
            w = sum(1.0 / label_counts[p] for p in s["problems"] if p in label_counts)
            return w if w > 0 else 1.0 / len(abnormal_idx)

        # Pool budgets: 40% for normals, 60% for abnormals
        normal_unit  = 0.4 / len(normal_idx)   if normal_idx   else 0.0
        abnormal_raw = {i: abnormal_weight(self.samples[i]) for i in abnormal_idx}
        total_raw    = sum(abnormal_raw.values())
        abnormal_budget = 0.6

        weights = [0.0] * len(self.samples)
        for i in normal_idx:
            weights[i] = normal_unit
        for i in abnormal_idx:
            weights[i] = abnormal_budget * abnormal_raw[i] / total_raw

        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        sample = self.samples[idx]
        report = sample["report"]

        # Sample a task and build target
        task = random.choice(self.tasks)
        target = build_target(task, report, sample["problems"])
        if target is None:
            # Skip: return next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))

        # Load and augment images
        views = prepare_views(
            sample["frontal"],
            sample["lateral"],
            drop_prob=self.drop_prob,
            augment=self.augment,
            image_size=self.image_size,
        )
        if not views:
            return self.__getitem__((idx + 1) % len(self.samples))

        # Build conversation in the format TRL/Qwen expects
        image_content = [{"type": "image", "image": img} for img in views]
        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": PROMPTS[task]}],
            },
            {
                "role": "assistant",
                "content": target,
            },
        ]

        return {
            "messages": messages,
            "task": task,
            "uid": report.uid,
        }
