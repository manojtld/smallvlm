"""
Load the Indiana University chest X-ray dataset.

Dataset root: /raid/manoj/smallvlm/raddar/chest-xrays-indiana-university/versions/2/
  indiana_reports.csv        - 3,851 reports (uid, MeSH, Problems, findings, impression, ...)
  indiana_projections.csv    - 7,466 rows (uid, filename, projection: Frontal|Lateral)
  images/images_normalized/  - 7,470 PNGs named {uid}_{series}.dcm.png

Notes:
  - 1,426 reports contain XXXX tokens (de-identified proper nouns) — pass through as-is.
  - 514 reports have null findings; impression is almost always present.
  - Some UIDs have multiple frontal or lateral images; we keep the first of each.
  - `Problems` is a cleaner version of `MeSH` (no anatomical qualifiers).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

# Default data directory — update via --data-dir CLI arg or DATA_DIR env var
DEFAULT_DATA_DIR = "/raid/manoj/smallvlm/raddar/chest-xrays-indiana-university/versions/2"


def load_projections(data_dir: Path) -> dict:
    """Return {uid: {frontal: path, lateral: path}} from indiana_projections.csv.

    When a UID has multiple frontal or lateral images (195 such UIDs), the first
    encountered is kept.
    """
    proj_path = data_dir / "indiana_projections.csv"
    if not proj_path.exists():
        return {}
    df = pd.read_csv(proj_path)
    image_dir = data_dir / "images" / "images_normalized"
    mapping: dict = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        projection = str(row.get("projection", "")).lower()
        filename = str(row.get("filename", ""))
        if not filename:
            continue
        full_path = str(image_dir / filename)
        if uid not in mapping:
            mapping[uid] = {}
        if "frontal" in projection or "pa" in projection:
            mapping[uid].setdefault("frontal", full_path)
        elif "lateral" in projection:
            mapping[uid].setdefault("lateral", full_path)
        else:
            mapping[uid].setdefault("frontal", full_path)
    return mapping


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_reports(data_dir: str | Path = DEFAULT_DATA_DIR) -> List[dict]:
    """
    Parse indiana_reports.csv and return a list of raw report dicts.

    Each dict contains:
      uid, raw_findings, raw_impression, mesh_tags, problems,
      indication, frontal_image, lateral_image
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "indiana_reports.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"indiana_reports.csv not found in {data_dir}.\n"
            f"Expected path: {DEFAULT_DATA_DIR}"
        )

    df = pd.read_csv(csv_path)
    projections = load_projections(data_dir)
    records = []
    for _, row in df.iterrows():
        uid = int(row["uid"])

        mesh_raw = _clean(row.get("MeSH", ""))
        mesh_tags = [t.strip() for t in mesh_raw.split(";") if t.strip()] if mesh_raw else []

        problems_raw = _clean(row.get("Problems", ""))
        problems = [t.strip() for t in problems_raw.split(";") if t.strip()] if problems_raw else []

        record = {
            "uid": uid,
            "raw_findings": _clean(row.get("findings", "")),
            "raw_impression": _clean(row.get("impression", "")),
            "mesh_tags": mesh_tags,
            "problems": problems,
            "indication": _clean(row.get("indication", "")),
            "frontal_image": projections.get(uid, {}).get("frontal"),
            "lateral_image": projections.get(uid, {}).get("lateral"),
        }
        records.append(record)

    n_frontal = sum(1 for r in records if r["frontal_image"])
    n_both = sum(1 for r in records if r["frontal_image"] and r["lateral_image"])
    print(f"Loaded {len(records)} reports — {n_frontal} with frontal image, {n_both} with both views")
    return records


def save_parsed(records: List[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} parsed records → {output_path}")


def load_parsed(path: str | Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
