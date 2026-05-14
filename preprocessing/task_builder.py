"""
Build SFT task triples from CanonicalReport objects.

Five task types, each generating a (prompt, target) pair per report:

  full_report           image → full findings text
  findings_only         image → bullet-list of findings
  impression_only       image → impression sentence
  structured_json       image → canonical JSON string
  normal_classification image → "Normal" | "Abnormal"

Output is a JSONL file of SFTTask objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .schema import CanonicalReport, SFTTask

TASK_PROMPTS = {
    "full_report": "Describe the radiological findings from this chest X-ray.",
    "findings_only": "List the individual findings from this chest X-ray.",
    "impression_only": "What is the clinical impression from this chest X-ray?",
    "structured_json": (
        "Analyze this chest X-ray and return a structured JSON report with fields: "
        "findings (list), impression (string), attributes (per-finding severity/location/size), "
        "and normal (boolean)."
    ),
    "normal_classification": "Is this chest X-ray normal or abnormal? Answer with a single word.",
}


def _build_full_report_target(report: CanonicalReport) -> Optional[str]:
    text = report.raw_findings or ""
    return text if text else None


def _build_findings_only_target(report: CanonicalReport) -> Optional[str]:
    if report.findings:
        return "\n".join(f"- {f}" for f in report.findings)
    if report.raw_findings:
        return report.raw_findings
    return None


def _build_impression_target(report: CanonicalReport) -> Optional[str]:
    text = report.impression or report.raw_impression
    return text if text else None


def _build_json_target(report: CanonicalReport) -> str:
    payload = {
        "findings": report.findings,
        "impression": report.impression,
        "attributes": {
            k: {"severity": v.severity, "location": v.location, "size": v.size}
            for k, v in report.attributes.items()
        },
        "normal": report.normal,
    }
    return json.dumps(payload)


def _build_normal_target(report: CanonicalReport) -> str:
    return "Normal" if report.normal else "Abnormal"


def build_tasks(
    reports: List[CanonicalReport],
    task_types: Optional[List[str]] = None,
) -> List[SFTTask]:
    """Generate SFT tasks for all reports. Skips tasks where target is empty."""
    if task_types is None:
        task_types = list(TASK_PROMPTS.keys())

    tasks: List[SFTTask] = []
    skipped = 0

    for report in reports:
        for task_type in task_types:
            if task_type == "full_report":
                target = _build_full_report_target(report)
            elif task_type == "findings_only":
                target = _build_findings_only_target(report)
            elif task_type == "impression_only":
                target = _build_impression_target(report)
            elif task_type == "structured_json":
                target = _build_json_target(report)
            elif task_type == "normal_classification":
                target = _build_normal_target(report)
            else:
                continue

            if not target:
                skipped += 1
                continue

            tasks.append(SFTTask(
                uid=report.uid,
                task_type=task_type,
                frontal_image=None,  # populated by attach_images()
                lateral_image=None,
                prompt=TASK_PROMPTS[task_type],
                target=target,
            ))

    print(f"Built {len(tasks)} tasks ({skipped} skipped due to empty targets)")
    return tasks


def attach_images(tasks: List[SFTTask], projections: dict) -> List[SFTTask]:
    """Attach frontal/lateral image paths from the projections mapping."""
    for task in tasks:
        proj = projections.get(task.uid, {})
        task.frontal_image = proj.get("frontal")
        task.lateral_image = proj.get("lateral")
    return tasks


def save_tasks(tasks: List[SFTTask], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")
    print(f"Saved {len(tasks)} SFT tasks → {output_path}")


def load_tasks(path: str | Path) -> List[SFTTask]:
    with open(path) as f:
        return [SFTTask.model_validate_json(line) for line in f if line.strip()]


def task_summary(tasks: List[SFTTask]) -> None:
    from collections import Counter
    counts = Counter(t.task_type for t in tasks)
    has_image = sum(1 for t in tasks if t.frontal_image)
    print(f"\nTask summary ({len(tasks)} total, {has_image} with frontal image):")
    for task_type, count in sorted(counts.items()):
        print(f"  {task_type:30s} {count:5d}")
