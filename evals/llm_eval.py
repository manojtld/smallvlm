"""
LLM-based radiology report evaluation.

Compares a model-generated report (findings + impression) against the ground
truth report using an LLM judge. Produces two scores per sample:

  accuracy (0-10)
    Measures correctness of pathology identification.
    - 10: All significant and minor findings correctly identified
    - 7-9: Trivial omissions or minor false positives
    - 4-6: Missing 1 significant pathology OR 1-2 significant false positives
    - 1-3: Multiple significant errors, wrong laterality on key finding
    - 0: Major critical error (e.g. calls clear pneumothorax/mass "normal")
    Penalty scale: significant finding missed > minor finding missed > false positive

  completeness (0-10)
    Measures quality and thoroughness of abnormality descriptions.
    - 10: Full description with location, severity, character for all findings
    - 7-9: Good descriptions, minor qualifiers missing
    - 4-6: Findings named but descriptions sparse
    - 1-3: Only finding names, no descriptive detail
    - 0: No findings or completely uninformative

Model: qwen/qwen3.6-plus (stronger than flash for judgment tasks)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from portkey_ai import Portkey

MODEL = os.environ.get("PORTKEY_EVAL_MODEL", "qwen/qwen3.6-plus")

JUDGE_PROMPT = """\
You are an expert radiologist evaluating the quality of an AI-generated chest X-ray report \
against a ground truth report written by a radiologist.

Score the AI report on two dimensions. Output ONLY valid JSON — no prose, no markdown fences.

## Ground Truth Report
FINDINGS: {gt_findings}
IMPRESSION: {gt_impression}

## AI-Generated Report
FINDINGS: {pred_findings}
IMPRESSION: {pred_impression}

## Scoring Rubric

### accuracy (0-10): Correctness of pathology identification
- 10: All significant and minor findings correctly identified, no false positives
- 8-9: Trivial omissions (incidental minor findings) OR 1 minor false positive
- 6-7: 1 significant pathology missed OR 2+ minor findings missed
- 4-5: 1 significant pathology missed AND false positives present
- 2-3: Multiple significant pathologies missed or wrong
- 0-1: Critical error — major pathology completely missed or wrongly invented
Significant pathologies (higher penalty if missed): pneumothorax, large effusion, \
consolidation, mass, cardiomegaly, pulmonary edema, fractures.
Minor/incidental (lower penalty if missed): calcified granuloma, mild scoliosis, \
minor osteophytes, small calcinosis.

### completeness (0-10): Thoroughness of abnormality descriptions
- 10: Every abnormality described with location, severity, and character
- 7-9: Good descriptions, missing some qualifiers (e.g. no laterality given)
- 4-6: Findings named but descriptions lack detail
- 1-3: Only named findings with no descriptive detail
- 0: No findings stated or completely uninformative output

## Output Schema
{{
  "accuracy_score": <0-10>,
  "completeness_score": <0-10>,
  "accuracy_reasoning": "<one sentence explaining the accuracy score>",
  "completeness_reasoning": "<one sentence explaining the completeness score>"
}}

Important: Judge based on clinical content, not wording. A normal study correctly \
identified as normal should score 10/10 on accuracy even if worded differently.
If the AI output is empty or garbled, score 0 on both dimensions.
"""


def _make_client() -> Portkey:
    return Portkey(
        api_key=os.environ["PORTKEY_API_KEY"],
        virtual_key=os.environ["PORTKEY_VIRTUAL_KEY"],
    )


def _score_one(
    uid: int,
    gt_findings: str,
    gt_impression: str,
    pred_findings: str,
    pred_impression: str,
    client: Portkey,
) -> Dict:
    prompt = JUDGE_PROMPT.format(
        gt_findings=gt_findings or "(none)",
        gt_impression=gt_impression or "(none)",
        pred_findings=pred_findings or "(empty)",
        pred_impression=pred_impression or "(empty)",
    )

    delay = 5
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            return {
                "uid": uid,
                "accuracy_score": float(data.get("accuracy_score", 0)),
                "completeness_score": float(data.get("completeness_score", 0)),
                "accuracy_reasoning": data.get("accuracy_reasoning", ""),
                "completeness_reasoning": data.get("completeness_reasoning", ""),
                "error": None,
            }
        except Exception as e:
            err = str(e)
            if attempt == 5:
                return {"uid": uid, "accuracy_score": None, "completeness_score": None,
                        "accuracy_reasoning": "", "completeness_reasoning": "", "error": err}
            if "429" in err or "500" in err or "rate" in err.lower() or "server" in err.lower():
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                return {"uid": uid, "accuracy_score": None, "completeness_score": None,
                        "accuracy_reasoning": "", "completeness_reasoning": "", "error": err}


def run_llm_eval(
    samples: List[Dict],
    checkpoint_path: Optional[str | Path] = None,
    workers: int = 15,
) -> List[Dict]:
    """
    Score each sample. Each dict must have:
      uid, gt_findings, gt_impression, pred_findings, pred_impression

    Checkpoints after each result for resumability.
    """
    client = _make_client()

    done: Dict[int, Dict] = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("accuracy_score") is not None:
                        done[r["uid"]] = r
        print(f"Resuming LLM eval: {len(done)} already scored")

    pending = [s for s in samples if s["uid"] not in done]
    if not pending:
        return list(done.values())

    ckpt_file = open(checkpoint_path, "a") if checkpoint_path else None
    results = list(done.values())

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _score_one,
                    s["uid"], s["gt_findings"], s["gt_impression"],
                    s["pred_findings"], s["pred_impression"], client
                ): s for s in pending
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if ckpt_file:
                    ckpt_file.write(json.dumps(result) + "\n")
                    ckpt_file.flush()
                completed += 1
                if completed % 50 == 0 or completed == len(pending):
                    print(f"  {completed}/{len(pending)} scored")
    finally:
        if ckpt_file:
            ckpt_file.close()

    return results


def summarize(results: List[Dict]) -> Dict:
    valid = [r for r in results if r.get("accuracy_score") is not None]
    if not valid:
        return {}
    acc  = [r["accuracy_score"] for r in valid]
    comp = [r["completeness_score"] for r in valid]
    return {
        "n": len(valid),
        "accuracy_mean":      round(sum(acc) / len(acc), 3),
        "accuracy_median":    round(sorted(acc)[len(acc) // 2], 3),
        "completeness_mean":  round(sum(comp) / len(comp), 3),
        "completeness_median":round(sorted(comp)[len(comp) // 2], 3),
        "accuracy_dist":      {str(i): sum(1 for s in acc  if int(s) == i) for i in range(11)},
        "completeness_dist":  {str(i): sum(1 for s in comp if int(s) == i) for i in range(11)},
    }
