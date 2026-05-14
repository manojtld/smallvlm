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

- Source: Kaggle dataset `raddar/chest-xrays-indiana-university`
- Downloaded via `kagglehub.dataset_download()`, which caches to `~/.cache/kagglehub/`
- Contains `indiana_reports.csv` (uid, findings, impression, MeSH), `indiana_projections.csv`, and `images/images_normalized/*.png`
- ~3,851 reports; each report may have a frontal and a lateral PNG

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

Requires `ANTHROPIC_API_KEY` env var for the format step.

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
