"""
CLI for the CXR preprocessing pipeline.

Steps:
  parse   - Load indiana_reports.csv → data/parsed.jsonl
  format  - LLM-format parsed reports → data/canonical.jsonl
  build   - Build SFT tasks from canonical reports → data/sft_tasks.jsonl
  all     - Run parse → format → build in sequence

Usage:
  python -m preprocessing.run parse  --data-dir /path/to/extracted/dataset
  python -m preprocessing.run format --input data/parsed.jsonl --output data/canonical.jsonl [--sync] [--limit 50]
  python -m preprocessing.run build  --input data/canonical.jsonl --data-dir /path/to/data --output data/sft_tasks.jsonl
  python -m preprocessing.run all    --data-dir /path/to/extracted/dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path("data")


def cmd_parse(args: argparse.Namespace) -> None:
    from .report_parser import load_reports, save_parsed
    records = load_reports(args.data_dir)
    save_parsed(records, args.output)


def cmd_format(args: argparse.Namespace) -> None:
    from .report_parser import load_parsed
    from .llm_formatter import format_reports, save_canonical

    records = load_parsed(args.input)
    if args.limit:
        records = records[: args.limit]
        print(f"Limiting to {args.limit} records")

    checkpoint = Path(args.output).with_suffix(".checkpoint.jsonl")
    reports = format_reports(records, checkpoint_path=checkpoint, workers=args.workers)
    save_canonical(reports, args.output)


def cmd_primitives(args: argparse.Namespace) -> None:
    from .llm_formatter import add_primitives, load_canonical, save_canonical
    reports = load_canonical(args.input)
    checkpoint = Path(args.output).with_suffix(".primitives_checkpoint.jsonl")
    reports = add_primitives(reports, checkpoint_path=checkpoint, workers=args.workers)
    save_canonical(reports, args.output)


def cmd_build(args: argparse.Namespace) -> None:
    from .llm_formatter import load_canonical
    from .report_parser import load_projections
    from .task_builder import attach_images, build_tasks, save_tasks, task_summary

    reports = load_canonical(args.input)
    projections = load_projections(Path(args.data_dir)) if args.data_dir else {}

    task_types = args.tasks.split(",") if args.tasks else None
    tasks = build_tasks(reports, task_types=task_types)

    if projections:
        tasks = attach_images(tasks, projections)

    save_tasks(tasks, args.output)
    task_summary(tasks)


def cmd_all(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    parsed = DATA_DIR / "parsed.jsonl"
    canonical = DATA_DIR / "canonical.jsonl"
    sft = DATA_DIR / "sft_tasks.jsonl"

    # parse
    parse_args = argparse.Namespace(data_dir=data_dir, output=parsed)
    cmd_parse(parse_args)

    # format
    format_args = argparse.Namespace(
        input=parsed,
        output=canonical,
        limit=args.limit,
        workers=args.workers,
    )
    cmd_format(format_args)

    # build
    build_args = argparse.Namespace(
        input=canonical,
        data_dir=data_dir,
        output=sft,
        tasks=None,
    )
    cmd_build(build_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="CXR preprocessing pipeline")
    subs = parser.add_subparsers(dest="command", required=True)

    # parse
    p_parse = subs.add_parser("parse", help="Load raw dataset → parsed.jsonl")
    p_parse.add_argument("--data-dir", required=True, help="Extracted dataset directory")
    p_parse.add_argument("--output", default=DATA_DIR / "parsed.jsonl")

    # format
    p_fmt = subs.add_parser("format", help="LLM-format parsed reports → canonical.jsonl")
    p_fmt.add_argument("--input", default=DATA_DIR / "parsed.jsonl")
    p_fmt.add_argument("--output", default=DATA_DIR / "canonical.jsonl")
    p_fmt.add_argument("--limit", type=int, default=None, help="Process only first N records")
    p_fmt.add_argument("--workers", type=int, default=10, help="Parallel threads")

    # primitives
    p_prim = subs.add_parser("primitives", help="Add primitive_observations to canonical.jsonl")
    p_prim.add_argument("--input",   default=DATA_DIR / "canonical.jsonl")
    p_prim.add_argument("--output",  default=DATA_DIR / "canonical.jsonl")
    p_prim.add_argument("--workers", type=int, default=15)

    # build
    p_build = subs.add_parser("build", help="Build SFT tasks from canonical reports")
    p_build.add_argument("--input", default=DATA_DIR / "canonical.jsonl")
    p_build.add_argument("--data-dir", default=None, help="Dataset dir for image paths")
    p_build.add_argument("--output", default=DATA_DIR / "sft_tasks.jsonl")
    p_build.add_argument("--tasks", default=None, help="Comma-separated task types to include")

    # all
    p_all = subs.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--data-dir", required=True)
    p_all.add_argument("--limit", type=int, default=None)
    p_all.add_argument("--workers", type=int, default=10)

    args = parser.parse_args()
    {"parse": cmd_parse, "format": cmd_format, "primitives": cmd_primitives,
     "build": cmd_build, "all": cmd_all}[args.command](args)


if __name__ == "__main__":
    main()
