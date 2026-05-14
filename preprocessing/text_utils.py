"""
Clean de-identification redactions from IU CXR report text.

The dataset replaces sensitive tokens with XXXX. Two strategies:
  1. Known substitutions  - replace specific patterns with the correct word
     (e.g. "x-XXXX" → "x-ray" appears ~700 times)
  2. Drop remaining XXXX  - remove tokens and fix resulting whitespace/punctuation
  3. Drop administrative sentences from impressions - physician names, phone
     calls, timestamps add no clinical value and are mostly XXXX anyway
"""

from __future__ import annotations

import re


# ── Known substitutions (order matters — more specific first) ─────────────────

_SUBSTITUTIONS = [
    (r"x-XXXX", "x-ray"),
    (r"X-XXXX", "X-ray"),
]

# Sentence-level patterns that indicate administrative content with no clinical
# value. Applied only to impression (not findings).
_ADMIN_PATTERNS = re.compile(
    r"(discussed|telephone|Dr\.|p\.m\.|a\.m\.|receipt|technologist|"
    r"radiograph(?:s)? were reviewed|results were|was notified|"
    r"communicated|phone|pager)",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on . and newlines."""
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    return [p.strip() for p in parts if p.strip()]


# Sentences starting with these words (after XXXX removal) have lost their
# subject and should be dropped.
_DANGLING_START = re.compile(
    r"^(are|is|were|was|have|has|had|do|does|did|appear|appears|appeared)\b",
    re.IGNORECASE,
)


def _clean_tokens(text: str) -> str:
    """Apply substitutions, remove XXXX tokens, fix whitespace artifacts."""
    for pattern, replacement in _SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s*\bXXXX\b\s*", " ", text)
    text = re.sub(r",\s+\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def clean_findings(text: str) -> str:
    """Remove XXXX tokens from findings text, dropping sentences with no subject."""
    if not text:
        return text

    sentences = _split_sentences(text)
    cleaned = []
    for sent in sentences:
        sent = _clean_tokens(sent)
        if not sent:
            continue
        if _DANGLING_START.match(sent):
            continue
        cleaned.append(sent)

    return " ".join(cleaned)


def clean_impression(text: str) -> str:
    """
    Remove XXXX tokens and drop administrative sentences from impression text.
    Administrative sentences (physician communication records) have no clinical
    value and are almost entirely redacted anyway.
    """
    if not text:
        return text

    for pattern, replacement in _SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)

    sentences = _split_sentences(text)
    clinical = []
    for sent in sentences:
        # Drop sentences that are primarily XXXX or administrative
        xxxx_count = sent.count("XXXX")
        word_count = len(sent.split())
        if word_count > 0 and xxxx_count / word_count > 0.4:
            continue
        if _ADMIN_PATTERNS.search(sent):
            continue
        clinical.append(sent)

    cleaned = []
    for sent in clinical:
        sent = _clean_tokens(sent)
        if not sent:
            continue
        if _DANGLING_START.match(sent):
            continue
        cleaned.append(sent)

    return " ".join(cleaned)
