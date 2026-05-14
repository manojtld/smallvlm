# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**smallvlm** is an early-stage project for training small Vision Language Models (VLMs) on chest X-ray medical imaging data. The current code is a data ingestion starting point; the broader goal is to build a training pipeline on top of it.

## Running the Code

```bash
# Download the Indiana University chest X-ray dataset from Kaggle
python download.py
```

Requires `kagglehub` and valid Kaggle credentials configured locally (`~/.kaggle/kaggle.json`).

## Dataset

**Location**: `/raid/manoj/smallvlm/raddar/chest-xrays-indiana-university/versions/2/`

| File | Description |
|---|---|
| `indiana_reports.csv` | 3,851 reports — uid, MeSH, Problems, findings, impression, indication, comparison |
| `indiana_projections.csv` | 7,466 rows mapping uid → filename + projection (Frontal/Lateral) |
| `images/images_normalized/` | 7,470 PNGs named `{uid}_{series}.dcm.png` |

Key facts:
- 3,388 UIDs have both frontal+lateral; 301 frontal-only; 162 lateral-only
- 195 UIDs have 3-5 images (multiple frontal or lateral shots) — we keep the first of each
- 514 reports have null findings (13%); impression is almost always present
- 1,379 normal reports (35.8%) — identified by `MeSH == "normal"`
- 1,426 reports contain `XXXX` tokens (de-identified proper nouns) — passed through as-is
- `Problems` column = cleaner version of `MeSH` (no anatomical qualifiers)

## Preprocessing Pipeline

The `preprocessing/` folder converts raw reports into SFT training data via three steps:

```bash
# 1. Parse raw CSV into JSONL
python -m preprocessing.run parse --data-dir /path/to/extracted/dataset

# 2. LLM-format reports into canonical JSON (uses Claude Batches API)
python -m preprocessing.run format                    # batch mode (~$2-3 for full dataset)
python -m preprocessing.run format --sync --limit 20  # sync mode for testing

# 3. Build SFT task triples from canonical reports
python -m preprocessing.run build --data-dir /path/to/extracted/dataset

# Or run all steps at once
python -m preprocessing.run all --data-dir /path/to/extracted/dataset
```

Requires `PORTKEY_API_KEY` and `PORTKEY_VIRTUAL_KEY` env vars for the format step.
Uses `qwen/qwen3.6-flash` via Portkey → OpenRouter by default.
Override the model with `PORTKEY_MODEL=qwen/qwen3.6-plus` (or any OpenRouter model ID).

### Data flow

```
indiana_reports.csv
      ↓ report_parser.py
data/parsed.jsonl          (uid, raw_findings, raw_impression, mesh_tags)
      ↓ llm_formatter.py   (Claude Haiku 4.5, Batches API, prompt caching)
data/canonical.jsonl       (CanonicalReport: structured findings list, impression, attributes, normal flag)
      ↓ task_builder.py
data/sft_tasks.jsonl       (SFTTask: 5 task types × N reports)
```

### SFT task types

| task_type | prompt | target |
|---|---|---|
| `full_report` | Describe the radiological findings | raw findings text |
| `findings_only` | List the individual findings | bullet list |
| `impression_only` | What is the clinical impression | impression string |
| `structured_json` | Return structured JSON report | canonical JSON |
| `normal_classification` | Normal or abnormal? | "Normal" / "Abnormal" |

### Key modules

- `schema.py` — Pydantic models: `CanonicalReport`, `FindingAttributes`, `SFTTask`
- `report_parser.py` — loads CSV, maps projections to image paths
- `llm_formatter.py` — submits/polls Batches API; also has `format_reports_sync()` for small runs
- `task_builder.py` — derives task triples; `attach_images()` wires in image paths
- `run.py` — CLI entry point (`python -m preprocessing.run`)
