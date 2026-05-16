"""
Run LLM-based report evaluation on model traces.

Usage:
  python -m evals.run_llm_eval --traces evals/results/baseline_qwen35_2b.traces.jsonl
  python -m evals.run_llm_eval --traces evals/results/sft_phase5_2b_5ep.traces.jsonl

Joins l3_findings/l3_impression from traces with GT from canonical.jsonl,
then calls the LLM judge and saves scored results + summary.

Output files (same dir as traces):
  <name>.llm_scores.jsonl  — per-sample scores and reasoning
  <name>.llm_summary.json  — aggregate stats
  <name>.llm_summary.txt   — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .llm_eval import run_llm_eval, summarize

CANONICAL_PATH = Path(os.environ.get("SMALLVLM_DATA", "/raid3/manoj/smallvlm")) / "data/canonical.jsonl"


def load_gt(canonical_path: Path) -> dict:
    """Return {uid: {findings, impression}} from canonical.jsonl."""
    gt = {}
    with open(canonical_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            uid = r["uid"]
            findings = " ".join(r.get("findings", [])) or r.get("raw_findings", "")
            impression = r.get("impression", "") or r.get("raw_impression", "")
            gt[uid] = {"findings": findings, "impression": impression}
    return gt


def load_traces(traces_path: Path) -> list:
    samples = []
    with open(traces_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_summary(model_name: str, summary: dict, traces_path: Path) -> str:
    lines = [
        "=" * 62,
        f"  LLM Eval — {model_name}",
        f"  Traces : {traces_path.name}",
        f"  Scored : {summary['n']} samples",
        "=" * 62,
        "",
        f"  Accuracy    (0-10)   mean={summary['accuracy_mean']:.2f}   "
        f"median={summary['accuracy_median']:.1f}",
        f"  Completeness(0-10)   mean={summary['completeness_mean']:.2f}   "
        f"median={summary['completeness_median']:.1f}",
        "",
        "  Accuracy distribution:",
        "  " + "  ".join(f"{k}:{v:3d}" for k, v in sorted(summary["accuracy_dist"].items(), key=lambda x: int(x[0]))),
        "",
        "  Completeness distribution:",
        "  " + "  ".join(f"{k}:{v:3d}" for k, v in sorted(summary["completeness_dist"].items(), key=lambda x: int(x[0]))),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, help="Path to .traces.jsonl file")
    parser.add_argument("--canonical", default=str(CANONICAL_PATH))
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--model-name", default=None,
                        help="Display name for the model (default: inferred from filename)")
    args = parser.parse_args()

    traces_path = Path(args.traces)
    out_dir = traces_path.parent
    stem = traces_path.name.replace(".traces.jsonl", "")
    model_name = args.model_name or stem

    scores_path  = out_dir / f"{stem}.llm_scores.jsonl"
    summary_json = out_dir / f"{stem}.llm_summary.json"
    summary_txt  = out_dir / f"{stem}.llm_summary.txt"
    ckpt_path    = out_dir / f"{stem}.llm_scores.checkpoint.jsonl"

    print(f"Loading GT from {args.canonical} ...")
    gt = load_gt(Path(args.canonical))
    print(f"Loading traces from {traces_path} ...")
    traces = load_traces(traces_path)

    # Build scored samples list
    samples = []
    skipped = 0
    for t in traces:
        uid = t["uid"]
        g = gt.get(uid, {})
        pred_findings  = t.get("l3_findings", "")
        pred_impression = t.get("l3_impression", "")
        if not pred_findings and not pred_impression:
            skipped += 1
        samples.append({
            "uid":             uid,
            "gt_findings":     g.get("findings", ""),
            "gt_impression":   g.get("impression", ""),
            "pred_findings":   pred_findings,
            "pred_impression": pred_impression,
        })

    print(f"Samples to score: {len(samples)}  ({skipped} with empty predictions)")

    results = run_llm_eval(samples, checkpoint_path=ckpt_path, workers=args.workers)

    # Save per-sample scores
    results.sort(key=lambda r: r["uid"])
    with open(scores_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved scores → {scores_path}")

    # Save summary
    s = summarize(results)
    summary_json.write_text(json.dumps(s, indent=2))

    txt = format_summary(model_name, s, traces_path)
    summary_txt.write_text(txt)
    print(f"\n{txt}")
    print(f"\nSaved → {summary_json}")
    print(f"Saved → {summary_txt}")


if __name__ == "__main__":
    main()
