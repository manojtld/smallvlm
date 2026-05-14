"""
Convert raw CXR reports into CanonicalReport JSON using Claude.

Two modes:
  - batch  (default): uses the Batches API for 50% cost reduction (~$1-3 for the full dataset)
  - sync           : direct calls, useful for testing a handful of records

Model: claude-haiku-4-5 by default (cheapest; adequate for structured extraction).
Change MODEL to claude-opus-4-7 for higher fidelity if budget allows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .schema import CanonicalReport, FindingAttributes

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """\
You are a radiology AI assistant. Convert the provided chest X-ray report into \
structured JSON. Output ONLY valid JSON — no prose, no markdown fences.

Schema:
{
  "findings": ["<finding as a short noun phrase>", ...],
  "impression": "<clinical impression as a single string>",
  "attributes": {
    "<finding>": {
      "severity": "<mild|moderate|severe|null>",
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
- Use null (JSON) not the string "null" for missing attributes.
"""


def _parse_llm_output(text: str, uid: int, raw: dict) -> CanonicalReport:
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
        # Fallback: keep raw text, mark as unparsed
        return CanonicalReport(
            uid=uid,
            mesh_tags=raw.get("mesh_tags", []),
            raw_findings=raw.get("raw_findings", ""),
            raw_impression=raw.get("raw_impression", ""),
        )


def _make_user_content(record: dict) -> str:
    findings = record.get("raw_findings") or "none"
    impression = record.get("raw_impression") or "none"
    return f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"


# ── Batch mode ────────────────────────────────────────────────────────────────

def submit_batch(records: List[dict], client: Optional[anthropic.Anthropic] = None) -> str:
    """Submit all records as a single batch. Returns batch_id."""
    client = client or anthropic.Anthropic()
    requests = [
        Request(
            custom_id=str(r["uid"]),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": _make_user_content(r)}],
            ),
        )
        for r in records
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}  ({len(requests)} requests)")
    return batch.id


def poll_batch(batch_id: str, poll_interval: int = 60, client: Optional[anthropic.Anthropic] = None) -> None:
    """Block until the batch completes."""
    client = client or anthropic.Anthropic()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            print(f"Batch {batch_id} complete — succeeded: {batch.request_counts.succeeded}, "
                  f"errored: {batch.request_counts.errored}")
            return
        remaining = batch.request_counts.processing
        print(f"Batch {batch_id}: {remaining} remaining … (sleeping {poll_interval}s)")
        time.sleep(poll_interval)


def collect_batch_results(
    batch_id: str,
    records_by_uid: dict,
    client: Optional[anthropic.Anthropic] = None,
) -> List[CanonicalReport]:
    """Collect and parse batch results. records_by_uid maps uid -> raw record."""
    client = client or anthropic.Anthropic()
    results = []
    for result in client.messages.batches.results(batch_id):
        uid = int(result.custom_id)
        raw = records_by_uid.get(uid, {"uid": uid})
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            results.append(_parse_llm_output(text, uid, raw))
        else:
            results.append(CanonicalReport(
                uid=uid,
                mesh_tags=raw.get("mesh_tags", []),
                raw_findings=raw.get("raw_findings", ""),
                raw_impression=raw.get("raw_impression", ""),
            ))
    return results


def format_reports_batch(
    records: List[dict],
    batch_id_file: Optional[str | Path] = None,
    poll_interval: int = 60,
) -> List[CanonicalReport]:
    """
    Full batch pipeline: submit → poll → collect.
    Saves the batch_id to batch_id_file so you can resume if interrupted.
    """
    client = anthropic.Anthropic()
    records_by_uid = {r["uid"]: r for r in records}

    if batch_id_file and Path(batch_id_file).exists():
        batch_id = Path(batch_id_file).read_text().strip()
        print(f"Resuming batch {batch_id}")
    else:
        batch_id = submit_batch(records, client)
        if batch_id_file:
            Path(batch_id_file).write_text(batch_id)

    poll_batch(batch_id, poll_interval, client)
    return collect_batch_results(batch_id, records_by_uid, client)


# ── Sync mode (testing / small subsets) ──────────────────────────────────────

def format_reports_sync(records: List[dict]) -> List[CanonicalReport]:
    """Process records one-by-one synchronously. Use for testing only."""
    client = anthropic.Anthropic()
    results = []
    for i, record in enumerate(records):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": _make_user_content(record)}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        results.append(_parse_llm_output(text, record["uid"], record))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(records)} done")
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
