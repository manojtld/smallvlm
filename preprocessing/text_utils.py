"""
Clean IU CXR report text before LLM processing.

Three passes:
  1. Known substitutions     (e.g. "x-XXXX" → "x-ray")
  2. XXXX redaction removal  (drop tokens, fix punctuation artifacts)
  3. Sentence-level filters  (drop admin sentences, pure-comparison sentences,
                              dangling predicates; strip leading temporal qualifiers)
"""

from __future__ import annotations

import re


# ── Substitutions ─────────────────────────────────────────────────────────────

_SUBSTITUTIONS = [
    (r"x-XXXX", "x-ray"),
    (r"X-XXXX", "X-ray"),
]

# ── Sentence-level drop patterns ──────────────────────────────────────────────

# Administrative — no clinical value (impression only)
_ADMIN_PATTERNS = re.compile(
    r"(discussed|telephone|Dr\.|p\.m\.|a\.m\.|receipt|technologist|"
    r"radiograph(?:s)? were reviewed|results were|was notified|"
    r"communicated|phone|pager)",
    re.IGNORECASE,
)

# Pure-comparison sentences — no clinical finding embedded, only temporal status.
# Safe to drop entirely since the VLM has no access to prior images.
_PURE_COMPARISON = re.compile(
    r"^("
    r"no interval change|no change(?! in)|unchanged appearance|stable appearance|"
    r"stable chest|no acute change|no significant change|no significant interval change|"
    r"no significant interval|interval change|compared to (?:prior|previous|the previous|old)|"
    r"comparison[:\s]"
    r")",
    re.IGNORECASE,
)

# Dangling predicates — subject was entirely redacted by XXXX removal
_DANGLING_START = re.compile(
    r"^(are|is|were|was|have|has|had|do|does|did|appear|appears|appeared)\b",
    re.IGNORECASE,
)

# ── Inline temporal qualifier stripping ───────────────────────────────────────

# Leading temporal qualifiers at the start of a sentence that can be stripped
# while preserving the clinical finding they describe.
# e.g. "Stable cardiomegaly." → "Cardiomegaly."
#      "Again noted is pleural effusion." → "Pleural effusion."
#      "Interval development of right lower lobe infiltrate." → "Right lower lobe infiltrate."
_LEADING_TEMPORAL = re.compile(
    r"^(?:"
    r"(?:stable|unchanged|persistent|redemonstrated?|re-demonstrated?)\s+"
    r"|again\s+(?:noted|observed|seen|demonstrated)\s+(?:is|are|was|were)?\s*"
    r"|(?:interval\s+)?(?:development|worsening|improvement|increase|decrease|progression)\s+of\s+"
    r"|in\s+the\s+interval[,\s]+(?:a\s+|an\s+|the\s+)?"
    r"|new\s+since\s+(?:prior|previous|last|the\s+prior|the\s+previous)\s+(?:exam|examination|study|x-ray|radiograph)?,?\s*"
    r")",
    re.IGNORECASE,
)

# Inline comparison phrases — stripped while preserving the clinical finding
# e.g. "cardiomegaly, stable compared to prior" → "cardiomegaly"
#      "normal in size and unchanged from prior examinations" → "normal in size"
_INLINE_COMPARISON = re.compile(
    r",?\s*(?:stable|unchanged|similar|redemonstrated?)\s+(?:compared\s+to\s+)?(?:prior|previous|the\s+prior|the\s+previous|old)?\s*(?:exam(?:ination)?|study|x-ray|radiograph|chest)?\s*(?:examination)?",
    re.IGNORECASE,
)
_AND_TEMPORAL = re.compile(
    r"\s+and\s+(?:stable|unchanged|similar|persistent)\s+(?:compared\s+to|from|since)?\s*(?:prior|previous|the\s+prior|the\s+previous|old)?[\w\s]*",
    re.IGNORECASE,
)
_COMPARED_TO = re.compile(
    r",?\s*compared\s+to\s+(?:prior|previous|the\s+prior|the\s+previous|old)[\w\s]*",
    re.IGNORECASE,
)
_FROM_PRIOR = re.compile(
    r",?\s*(?:and\s+)?(?:unchanged|stable)?\s*(?:from|since|on)\s+(?:prior|previous|the\s+prior|the\s+previous|old)\s+(?:exam(?:ination)?s?|study|x-ray|radiograph)?",
    re.IGNORECASE,
)

# Leading conjunctions left after temporal stripping ("And left midlung...")
_LEADING_CONJUNCTION = re.compile(r"^(and|but|or|with)\s+", re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    return [p.strip() for p in parts if p.strip()]


def _clean_tokens(text: str) -> str:
    for pattern, replacement in _SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s*\bXXXX\b\s*", " ", text)
    text = re.sub(r",\s+\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def _strip_temporal(sent: str) -> str:
    """Remove leading temporal qualifiers and inline comparison phrases."""
    sent = _AND_TEMPORAL.sub("", sent)
    sent = _FROM_PRIOR.sub("", sent)
    sent = _INLINE_COMPARISON.sub("", sent)
    sent = _COMPARED_TO.sub("", sent)
    sent = _LEADING_TEMPORAL.sub("", sent)
    sent = _LEADING_CONJUNCTION.sub("", sent)
    sent = sent.strip(" ,;")
    if sent:
        sent = sent[0].upper() + sent[1:]
    return sent


def _process_sentence(sent: str) -> str | None:
    """Clean a single sentence. Returns None if the sentence should be dropped."""
    sent = _clean_tokens(sent)
    if not sent:
        return None
    if _DANGLING_START.match(sent):
        return None
    if _PURE_COMPARISON.match(sent):
        return None
    sent = _strip_temporal(sent)
    if not sent:
        return None
    # Drop sentences that became meaninglessly short after stripping
    # (e.g. "Vascular appearance." with no qualifier)
    words = sent.rstrip(".").split()
    if len(words) <= 2 and not any(c.isalpha() for c in sent[2:]):
        return None
    return sent


def clean_findings(text: str) -> str:
    if not text:
        return text
    sentences = _split_sentences(text)
    cleaned = [s for sent in sentences if (s := _process_sentence(sent))]
    return " ".join(cleaned)


def clean_impression(text: str) -> str:
    if not text:
        return text

    sentences = _split_sentences(text)
    filtered = []
    for sent in sentences:
        xxxx_count = sent.count("XXXX")
        word_count = len(sent.split())
        if word_count > 0 and xxxx_count / word_count > 0.4:
            continue
        if _ADMIN_PATTERNS.search(sent):
            continue
        filtered.append(sent)

    cleaned = [s for sent in filtered if (s := _process_sentence(sent))]
    return " ".join(cleaned)
