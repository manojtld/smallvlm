"""
Convert raw CXR reports into CanonicalReport JSON using Qwen via Portkey.

Requires env vars:
  PORTKEY_API_KEY       - your Portkey API key
  PORTKEY_VIRTUAL_KEY   - virtual key pointing to OpenRouter (e.g. openrouter-f6f680)

Model: qwen/qwen3.6-flash by default (fast, cheap, 1M context).
Set MODEL env var or pass model= to override.

Processing uses a thread pool for parallelism and checkpoints results to disk
so runs can be resumed after interruption.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from portkey_ai import Portkey

from .schema import CanonicalReport, FindingAttributes

MODEL = os.environ.get("PORTKEY_MODEL", "qwen/qwen3.6-flash")

SYSTEM_PROMPT = """\
You are a radiology AI assistant. Convert the provided chest X-ray report into \
structured JSON. Output ONLY valid JSON — no prose, no markdown fences.

Schema:
{
  "findings": ["<finding as a short noun phrase>", ...],
  "impression": "<clinical impression as a single string>",
  "attributes": {
    "<finding>": {
      "severity": "<mild|moderate|severe or null>",
      "location": "<anatomical location or null>",
      "size": "<size description or null>"
    }
  },
  "normal": <true if explicitly normal, false otherwise>
}

Rules:
- Split compound findings into individual items (e.g. "cardiomegaly" and "pleural effusion" separately).
- Preserve the original impression text verbatim.
- Set normal=true only when the report states no abnormalities.
- Use null (JSON null, not the string "null") for missing attribute values.
"""


def _make_client() -> Portkey:
    return Portkey(
        api_key=os.environ["PORTKEY_API_KEY"],
        virtual_key=os.environ["PORTKEY_VIRTUAL_KEY"],
    )


def _make_user_content(record: dict) -> str:
    findings = record.get("raw_findings") or "none"
    impression = record.get("raw_impression") or "none"
    return f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"


def _parse_llm_output(text: str, uid: int, raw: dict) -> CanonicalReport:
    # Strip markdown fences if the model wraps output despite instructions
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        attributes = {
            k: FindingAttributes(**v) if isinstance(v, dict) else FindingAttributes()
            for k, v in data.get("attributes", {}).items()
        }
        return CanonicalReport(
            uid=uid,
            findings=data.get("findings", []),
            impression=data.get("impression", ""),
            attributes=attributes,
            normal=bool(data.get("normal", False)),
            mesh_tags=raw.get("mesh_tags", []),
            raw_findings=raw.get("raw_findings", ""),
            raw_impression=raw.get("raw_impression", ""),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return CanonicalReport(
            uid=uid,
            mesh_tags=raw.get("mesh_tags", []),
            raw_findings=raw.get("raw_findings", ""),
            raw_impression=raw.get("raw_impression", ""),
        )


def _format_one(record: dict, client: Portkey) -> CanonicalReport:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _make_user_content(record)},
        ],
    )
    text = response.choices[0].message.content or ""
    return _parse_llm_output(text, record["uid"], record)


# ── Main formatting function ──────────────────────────────────────────────────

def format_reports(
    records: List[dict],
    checkpoint_path: Optional[str | Path] = None,
    workers: int = 10,
) -> List[CanonicalReport]:
    """
    Format all records using a thread pool.

    Checkpoints results to `checkpoint_path` (JSONL) after each completed
    record so a run can be resumed after interruption — already-processed
    UIDs are skipped on restart.

    Args:
        records:         List of raw report dicts from report_parser.
        checkpoint_path: Path to a JSONL file for incremental saves.
                         Pass None to skip checkpointing.
        workers:         Number of parallel threads (default 10).
    """
    client = _make_client()

    # Load already-processed UIDs from checkpoint
    done_uids: set = set()
    results: List[CanonicalReport] = []
    if checkpoint_path and Path(checkpoint_path).exists():
        existing = load_canonical(checkpoint_path)
        done_uids = {r.uid for r in existing}
        results = existing
        print(f"Resuming: {len(done_uids)} already done, {len(records) - len(done_uids)} remaining")

    pending = [r for r in records if r["uid"] not in done_uids]
    if not pending:
        print("All records already processed.")
        return results

    checkpoint_file = open(checkpoint_path, "a") if checkpoint_path else None

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_format_one, r, client): r for r in pending}
            completed = 0
            for future in as_completed(futures):
                report = future.result()
                results.append(report)
                if checkpoint_file:
                    checkpoint_file.write(report.model_dump_json() + "\n")
                    checkpoint_file.flush()
                completed += 1
                if completed % 50 == 0 or completed == len(pending):
                    print(f"  {completed}/{len(pending)} done")
    finally:
        if checkpoint_file:
            checkpoint_file.close()

    return results


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_canonical(reports: List[CanonicalReport], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in reports:
            f.write(r.model_dump_json() + "\n")
    print(f"Saved {len(reports)} canonical reports → {output_path}")


def load_canonical(path: str | Path) -> List[CanonicalReport]:
    with open(path) as f:
        return [CanonicalReport.model_validate_json(line) for line in f if line.strip()]
