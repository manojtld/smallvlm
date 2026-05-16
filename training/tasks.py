"""
Task definitions for each SFT curriculum phase.

Each task produces a (prompt, target) pair given a CanonicalReport + Problems list.
The eval L2 prompt is reused verbatim for tag_classification so evals are comparable.
"""

from __future__ import annotations

import json
from typing import List, Optional

from evals.vocab import EVAL_LABELS

# ── Prompts ────────────────────────────────────────────────────────────────────

PROMPTS = {
    "primitive_observations": (
        "You are a radiologist. Describe the primitive visual observations visible "
        "in this chest X-ray using basic radiological descriptors only — no diagnoses.\n"
        "Output a JSON list of short observation strings."
    ),
    "tag_classification": (
        "You are a radiologist. Look at this chest X-ray.\n"
        "For each finding below, state whether it is present (true) or absent (false).\n"
        "Output ONLY a valid JSON object — no prose, no markdown fences.\n\n"
        "Findings:\n"
        + "\n".join(f"- {l}" for l in EVAL_LABELS)
        + '\n\nOutput format: {"Finding Name": true/false, ...}'
    ),
    "mesh_tags": (
        "You are a radiologist. List the MeSH (Medical Subject Heading) tags that "
        "describe the findings in this chest X-ray.\n"
        "Output a JSON list of tag strings."
    ),
    "findings": (
        "You are a radiologist. Describe all radiological findings visible in this "
        "chest X-ray as a concise list.\n"
        "Output each finding on a separate line."
    ),
    "impression": (
        "You are a radiologist. Write the clinical impression for this chest X-ray "
        "in 1-3 sentences. Include only the clinical conclusion — no administrative content."
    ),
    "structured_json": (
        "You are a radiologist. Analyze this chest X-ray and return a structured JSON report.\n"
        "Schema: {\"findings\": [...], \"impression\": \"...\", \"recommendation\": \"...\", "
        "\"pathology_json\": {\"<finding>\": {\"presence\": bool, \"location\": str|null, "
        "\"size\": str|null, \"texture\": str|null, \"prominence_score\": 0-5}}}\n"
        "Output ONLY valid JSON — no prose, no markdown fences."
    ),
}

# ── Curriculum phases ──────────────────────────────────────────────────────────

PHASES = {
    1: ["primitive_observations"],
    2: ["primitive_observations", "tag_classification"],
    3: ["primitive_observations", "tag_classification", "mesh_tags"],
    4: ["primitive_observations", "tag_classification", "mesh_tags", "findings", "impression"],
    5: ["primitive_observations", "tag_classification", "mesh_tags", "findings", "impression", "structured_json"],
    # Phase 6: findings+impression primary, classification secondary (with upweighted loss)
    6: ["findings", "findings", "impression", "impression", "tag_classification"],
}

# ── Target builders ────────────────────────────────────────────────────────────

def build_target(task: str, report, problems: List[str]) -> Optional[str]:
    """Return the ground-truth target string for a given task, or None if not available."""

    if task == "primitive_observations":
        if not report.primitive_observations:
            return None
        return json.dumps(report.primitive_observations)

    elif task == "tag_classification":
        problem_set = {p.lower() for p in problems}
        labels = {l: (l.lower() in problem_set) for l in EVAL_LABELS}
        return json.dumps(labels)

    elif task == "mesh_tags":
        if not report.mesh_tags:
            return None
        return json.dumps(report.mesh_tags)

    elif task == "findings":
        text = "\n".join(f"- {f}" for f in report.findings) if report.findings else report.raw_findings
        return text if text.strip() else None

    elif task == "impression":
        text = report.impression or report.raw_impression
        return text if text.strip() else None

    elif task == "structured_json":
        payload = {
            "findings": report.findings,
            "impression": report.impression,
            "recommendation": report.recommendation,
            "pathology_json": {
                k: {
                    "presence": v.presence,
                    "location": v.location,
                    "size": v.size,
                    "texture": v.texture,
                    "prominence_score": v.prominence_score,
                }
                for k, v in report.pathology_json.items()
            },
        }
        return json.dumps(payload)

    return None
