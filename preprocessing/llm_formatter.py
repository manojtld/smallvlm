"""
Convert raw CXR reports into CanonicalReport JSON using Qwen via Portkey.

Requires env vars:
  PORTKEY_API_KEY       - your Portkey API key
  PORTKEY_VIRTUAL_KEY   - virtual key pointing to OpenRouter (e.g. openrouter-f6f680)

Model: qwen/qwen3.6-flash by default (fast, cheap, 1M context).
Set PORTKEY_MODEL env var to override.

Processing uses a thread pool for parallelism and checkpoints results to disk
so runs can be resumed after interruption.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from portkey_ai import Portkey

from .schema import CanonicalReport, PathologyAttributes

MODEL = os.environ.get("PORTKEY_MODEL", "qwen/qwen3.6-flash")

SYSTEM_PROMPT = """\
You are a radiology AI assistant. Convert the provided chest X-ray report into \
structured JSON. Output ONLY valid JSON — no prose, no markdown fences.

Output schema:
{
  "findings": ["<finding as a short noun phrase>", ...],
  "impression": "<clinical conclusion only>",
  "recommendation": "<follow-up or further imaging recommendation, or empty string>",
  "pathology_json": {
    "<finding name>": {
      "presence": <true if present, false if explicitly absent>,
      "location": "<anatomical location or null>",
      "size": "<size description or null>",
      "texture": "<texture/character description or null>",
      "prominence_score": <0-5>
    }
  }
}

Prominence score guide (0-5):
  5 - Definite: stated as fact with no qualification ("cardiomegaly", "there is pleural effusion")
  4 - Likely: "consistent with", "findings suggest", "appears to be"
  3 - Probable: "likely", "probable", "probably"
  2 - Possible: "possible", "possibly", "may represent", "cannot exclude"
  1 - Questionable: "questionable", "differential includes", "consider", "rule out"
  0 - Absent: "no", "no evidence of", "without", "absent" — set presence=false

Rules for findings:
- Include ALL findings mentioned — both present and absent (e.g. "No pleural effusion" \
is a finding with presence=false, prominence_score=0).
- Split compound findings into individual items.
- The input may contain grammatically incomplete phrases where words were redacted \
(e.g. "There are no of a pleural effusion", "cardiac with leads"). \
Use clinical context to infer the intended meaning and write a clean, complete phrase \
(e.g. "No pleural effusion", "Cardiac device with leads").
- Only include findings with clear clinical meaning. Skip any phrase you cannot \
confidently interpret.
- The model only sees the current image — do not include any temporal or comparative \
language. If a finding is described as "stable", "unchanged", "new since prior", \
"interval worsening", "again noted", etc., extract only the finding itself based on \
its current appearance. Never use words like "stable", "unchanged", "interval", \
"prior", "previous", "again" in finding names or attributes.

Rules for impression:
- Include only the clinical conclusion — what the radiologist determined from the study.
- Exclude all administrative content: findings discussed with patient/physician, \
telephone communications, technologist notes, scheduling — anything non-clinical.

Rules for recommendation:
- Extract any explicit follow-up or imaging recommendation (e.g. "CT chest recommended", \
"clinical correlation suggested", "short interval follow-up").
- Leave as empty string if none is present.

General:
- Use null (JSON null, not the string "null") for missing attribute fields.
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
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        pathology_json = {}
        for k, v in data.get("pathology_json", {}).items():
            if isinstance(v, dict):
                # Clamp prominence_score to 0-5
                v["prominence_score"] = max(0, min(5, int(v.get("prominence_score", 5))))
                pathology_json[k] = PathologyAttributes(**v)
            else:
                pathology_json[k] = PathologyAttributes()

        return CanonicalReport(
            uid=uid,
            findings=data.get("findings", []),
            impression=data.get("impression", ""),
            recommendation=data.get("recommendation", ""),
            mesh_tags=raw.get("mesh_tags", []),
            pathology_json=pathology_json,
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
    delay = 5
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _make_user_content(record)},
                ],
            )
            text = response.choices[0].message.content or ""
            return _parse_llm_output(text, record["uid"], record)
        except Exception as e:
            if attempt == 5:
                raise
            err = str(e)
            if "429" in err or "500" in err or "rate" in err.lower() or "server" in err.lower():
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise


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
    """
    client = _make_client()

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
