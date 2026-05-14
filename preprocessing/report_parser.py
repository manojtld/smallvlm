"""
Load the Indiana University chest X-ray dataset from a local data directory.

Expected directory layout after extracting the Kaggle archive:
    <data_dir>/
        indiana_reports.csv
        indiana_projections.csv
        images/images_normalized/*.png
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

from .schema import CanonicalReport


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract the Kaggle zip archive if not already done."""
    if (dest_dir / "indiana_reports.csv").exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} → {dest_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Extraction complete.")


def load_projections(data_dir: Path) -> dict:
    """Return {uid: {frontal: path, lateral: path}} from indiana_projections.csv."""
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
            mapping[uid]["frontal"] = full_path
        elif "lateral" in projection:
            mapping[uid]["lateral"] = full_path
        else:
            mapping[uid].setdefault("frontal", full_path)
    return mapping


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_reports(data_dir: str | Path) -> List[dict]:
    """
    Parse indiana_reports.csv and return a list of raw report dicts.
    Each dict has: uid, raw_findings, raw_impression, mesh_tags.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "indiana_reports.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"indiana_reports.csv not found in {data_dir}. "
            "Run extract_archive() first or point --data-dir at the extracted folder."
        )

    df = pd.read_csv(csv_path)
    projections = load_projections(data_dir)
    records = []
    for _, row in df.iterrows():
        uid = int(row["uid"])
        mesh_raw = _clean(row.get("MeSH", ""))
        mesh_tags = [t.strip() for t in mesh_raw.split(";") if t.strip()] if mesh_raw else []
        record = {
            "uid": uid,
            "raw_findings": _clean(row.get("findings", "")),
            "raw_impression": _clean(row.get("impression", "")),
            "mesh_tags": mesh_tags,
            "frontal_image": projections.get(uid, {}).get("frontal"),
            "lateral_image": projections.get(uid, {}).get("lateral"),
        }
        records.append(record)
    print(f"Loaded {len(records)} reports from {csv_path}")
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
