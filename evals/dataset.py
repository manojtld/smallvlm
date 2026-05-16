"""
Load and split the IU CXR dataset for evaluation.

Produces a fixed stratified 80/20 train/test split (seed=42) saved to
evals/data/test_split.json so results are reproducible across runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(os.environ.get("SMALLVLM_DATA", "/raid/manoj/smallvlm")) / "raddar/chest-xrays-indiana-university/versions/2"
SPLIT_CACHE = Path(__file__).parent / "data" / "test_split.json"


@dataclass
class EvalSample:
    uid: int
    frontal_image: Optional[str]
    lateral_image: Optional[str]
    is_normal: bool                    # ground truth for Level 1
    problems: List[str] = field(default_factory=list)  # ground truth for Level 2


def _load_projections() -> dict:
    df = pd.read_csv(DATA_DIR / "indiana_projections.csv")
    image_dir = DATA_DIR / "images" / "images_normalized"
    mapping: dict = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        proj = str(row.get("projection", "")).lower()
        fname = str(row.get("filename", ""))
        if not fname:
            continue
        path = str(image_dir / fname)
        if uid not in mapping:
            mapping[uid] = {}
        if "frontal" in proj or "pa" in proj:
            mapping[uid].setdefault("frontal", path)
        elif "lateral" in proj:
            mapping[uid].setdefault("lateral", path)
        else:
            mapping[uid].setdefault("frontal", path)
    return mapping


def load_test_split(test_size: float = 0.2, seed: int = 42) -> List[EvalSample]:
    """
    Load the test split. Saves to SPLIT_CACHE on first call; reloads on subsequent calls.
    Stratified by normal/abnormal to preserve class balance.
    """
    SPLIT_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if SPLIT_CACHE.exists():
        test_uids = set(json.loads(SPLIT_CACHE.read_text()))
    else:
        df = pd.read_csv(DATA_DIR / "indiana_reports.csv")
        uids = df["uid"].tolist()
        is_normal = (df["MeSH"].fillna("") == "normal").tolist()
        _, test_uids_list = train_test_split(
            uids, test_size=test_size, random_state=seed, stratify=is_normal
        )
        test_uids = set(test_uids_list)
        SPLIT_CACHE.write_text(json.dumps(sorted(test_uids)))
        print(f"Created test split: {len(test_uids)} samples → {SPLIT_CACHE}")

    df = pd.read_csv(DATA_DIR / "indiana_reports.csv")
    projections = _load_projections()

    samples = []
    for _, row in df.iterrows():
        uid = int(row["uid"])
        if uid not in test_uids:
            continue
        problems_raw = str(row.get("Problems", "") or "")
        problems = [t.strip() for t in problems_raw.split(";") if t.strip() and t.strip().lower() != "normal"]
        proj = projections.get(uid, {})
        samples.append(EvalSample(
            uid=uid,
            frontal_image=proj.get("frontal"),
            lateral_image=proj.get("lateral"),
            is_normal=(str(row.get("MeSH", "")).strip() == "normal"),
            problems=problems,
        ))

    n_normal = sum(s.is_normal for s in samples)
    print(f"Test split: {len(samples)} samples ({n_normal} normal, {len(samples)-n_normal} abnormal)")
    return samples
