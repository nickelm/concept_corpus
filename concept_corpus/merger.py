"""
Concept merging.

After extraction, different documents may use different names for the same
concept ("situated visualization" vs "situated analytics" vs "in-situ display").
This module merges near-duplicates into canonical forms.

Two strategies:
1. Fast/free: string similarity + simple heuristics (no LLM needed)
2. Thorough: LLM-based grouping (one call per rebuild, not per document)
"""

import json
import re
from collections import Counter, defaultdict
from typing import Optional

from .models import ExtractedDocument, Concept


# ── String similarity merging (fast, free) ───────────────────

def _normalize(name: str) -> str:
    """Normalize a concept name for comparison."""
    name = name.lower().strip()
    # Remove trailing 's' for basic plurals
    if name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]
    # Remove common filler words
    for filler in ["based", "driven", "oriented", "aware"]:
        name = name.replace(f"-{filler}", f" {filler}")
    return name


def _tokenize(name: str) -> set[str]:
    """Split a concept name into word tokens."""
    return set(re.split(r"[\s\-_/]+", _normalize(name)))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def merge_by_similarity(
    docs: list[ExtractedDocument],
    threshold: float = 0.6,
) -> dict[str, str]:
    """
    Build a mapping from concept names to canonical forms using token overlap.

    Returns:
        Dict mapping original_name (lowercased) -> canonical_name.
        Concepts that don't merge map to themselves.
    """
    # Collect all concept names with their frequencies
    name_counts = Counter()
    for doc in docs:
        for c in doc.concepts:
            name_counts[c.name.lower().strip()] += 1

    names = list(name_counts.keys())
    tokens = {name: _tokenize(name) for name in names}

    # Group similar names
    # Use the most frequent name in each group as canonical
    merged: dict[str, str] = {}
    used = set()

    # Sort by frequency descending — most common name becomes canonical
    sorted_names = sorted(names, key=lambda n: name_counts[n], reverse=True)

    for name in sorted_names:
        if name in used:
            continue

        # This name becomes canonical for its group
        group = [name]
        used.add(name)

        for other in sorted_names:
            if other in used:
                continue
            sim = _jaccard(tokens[name], tokens[other])
            if sim >= threshold:
                group.append(other)
                used.add(other)

        # Map all group members to canonical name
        canonical = name  # most frequent
        for member in group:
            merged[member] = canonical

    return merged


# ── LLM-based merging (thorough, one call per rebuild) ───────

MERGE_PROMPT = """You are building a canonical concept index for a research corpus. Below is a list of concept names extracted from multiple academic papers. Many are near-duplicates or synonyms.

Group the concepts that refer to the same or very similar idea. For each group, pick the best canonical name — the most standard, widely-used term in the field.

Respond with ONLY valid JSON: a list of groups, where each group has a "canonical" name and a "members" list:
[
  {
    "canonical": "information visualization",
    "members": ["information visualization", "infovis", "data visualization", "visual data analysis"]
  },
  ...
]

Rules:
- Only group concepts that genuinely refer to the same thing. Don't merge "augmented reality" with "virtual reality."
- Singleton groups (concepts with no synonyms) can be omitted — they keep their original name.
- Use established terminology for the canonical name.

CONCEPTS:
"""


def merge_by_llm(
    docs: list[ExtractedDocument],
    model: str = "anthropic/claude-haiku-4-5-20251001",
    api_base: Optional[str] = None,
) -> dict[str, str]:
    """
    Use an LLM to group synonymous concepts and pick canonical names.

    This makes one LLM call per rebuild (not per document).

    Returns:
        Dict mapping original_name (lowercased) -> canonical_name.
    """
    import litellm

    # Collect all unique concept names
    all_names = set()
    for doc in docs:
        for c in doc.concepts:
            all_names.add(c.name.lower().strip())

    names_list = sorted(all_names)

    if len(names_list) < 2:
        return {n: n for n in names_list}

    # Build prompt
    names_text = "\n".join(f"- {n}" for n in names_list)

    kwargs = {
        "model": model,
        "messages": [
            {"role": "user", "content": MERGE_PROMPT + names_text}
        ],
        "temperature": 0.0,
        "max_tokens": 16384,
    }
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.completion(**kwargs)
    raw = response.choices[0].message.content.strip()

    # Parse JSON — handle markdown fences and truncation
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    # Try parsing as-is first
    try:
        groups = json.loads(raw)
    except json.JSONDecodeError:
        # Response may be truncated — try to recover by closing the JSON
        # Find the last complete group object
        last_brace = raw.rfind("}")
        if last_brace > 0:
            truncated = raw[:last_brace + 1] + "\n]"
            try:
                groups = json.loads(truncated)
            except json.JSONDecodeError:
                print("Warning: could not parse LLM merge response. Using fast merge as fallback.",
                      file=__import__('sys').stderr)
                return merge_by_similarity(docs)

    # Build mapping
    merged = {n: n for n in names_list}  # default: map to self
    for group in groups:
        canonical = group["canonical"].lower().strip()
        for member in group.get("members", []):
            member_lower = member.lower().strip()
            if member_lower in merged:
                merged[member_lower] = canonical

    return merged


# ── Apply merges to documents ────────────────────────────────

def apply_merges(
    docs: list[ExtractedDocument],
    merge_map: dict[str, str],
) -> list[ExtractedDocument]:
    """
    Apply a merge mapping to all documents, renaming concepts in place.

    Returns the same document objects with concept names updated.
    """
    for doc in docs:
        for concept in doc.concepts:
            original = concept.name.lower().strip()
            canonical = merge_map.get(original, original)
            concept.name = canonical
            # Also update related references
            concept.related = [
                merge_map.get(r.lower().strip(), r.lower().strip())
                for r in concept.related
            ]
    return docs


def deduplicate_within_doc(doc: ExtractedDocument) -> ExtractedDocument:
    """
    After merging, a document may have duplicate concepts.
    Keep only the first occurrence of each concept name.
    """
    seen = set()
    unique = []
    for c in doc.concepts:
        key = c.name.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    doc.concepts = unique
    return doc