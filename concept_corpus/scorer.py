"""
Distinctiveness scoring.

Computes a TF-IDF-style score for concepts, but at the semantic level:
- "Term frequency" = how prominent is this concept in the target set?
- "Inverse document frequency" = how rare is this concept in the reference set?

The target and reference are arbitrary subsets of the corpus, specified by
the caller. This keeps the scorer general — it doesn't know about
contributors, time windows, or document types.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .models import ExtractedDocument


@dataclass
class ConceptScore:
    """Score for a single concept's distinctiveness."""
    concept_name: str
    tf: float              # concept frequency in target (normalized)
    idf: float             # inverse document frequency in reference
    tf_idf: float          # combined score
    target_count: int      # raw count in target docs
    reference_count: int   # raw count in reference docs
    target_docs: int       # number of target docs containing this concept
    reference_docs: int    # number of reference docs containing this concept

    def to_dict(self) -> dict:
        return {
            "concept": self.concept_name,
            "score": round(self.tf_idf, 4),
            "tf": round(self.tf, 4),
            "idf": round(self.idf, 4),
            "in_target": self.target_docs,
            "in_reference": self.reference_docs,
        }


def distinctiveness(
    target: list[ExtractedDocument],
    reference: list[ExtractedDocument],
    min_target_count: int = 1,
    top_n: Optional[int] = None,
) -> list[ConceptScore]:
    """
    Compute concept distinctiveness of target relative to reference.

    High scores mean: concept is frequent in target, rare in reference.
    This is the semantic equivalent of TF-IDF.

    Args:
        target: Documents whose distinctive concepts we want to find.
        reference: Documents forming the baseline for comparison.
        min_target_count: Minimum times a concept must appear in target.
        top_n: If set, return only the top N most distinctive concepts.

    Returns:
        List of ConceptScore, sorted by tf_idf descending.
    """
    if not target:
        return []

    # Count concept occurrences in target (document frequency)
    target_df = Counter()      # concept -> number of target docs containing it
    target_tf_raw = Counter()  # concept -> total mentions across target docs
    for doc in target:
        seen = set()
        for c in doc.concepts:
            name = c.name.lower().strip()
            target_tf_raw[name] += 1
            if name not in seen:
                target_df[name] += 1
                seen.add(name)

    # Count concept occurrences in reference
    ref_df = Counter()
    for doc in reference:
        seen = set()
        for c in doc.concepts:
            name = c.name.lower().strip()
            if name not in seen:
                ref_df[name] += 1
                seen.add(name)

    n_target = len(target)
    n_ref = len(reference) if reference else 1  # avoid division by zero

    scores = []
    for concept_name, t_count in target_tf_raw.items():
        if t_count < min_target_count:
            continue

        t_doc_count = target_df[concept_name]
        r_doc_count = ref_df.get(concept_name, 0)

        # TF: proportion of target documents containing this concept
        tf = t_doc_count / n_target

        # IDF: log(N / (1 + df)) — how rare in the reference set
        # Adding 1 to denominator for smoothing (avoid log(inf) for unseen concepts)
        idf = math.log((n_ref + 1) / (1 + r_doc_count))

        tf_idf = tf * idf

        scores.append(ConceptScore(
            concept_name=concept_name,
            tf=tf,
            idf=idf,
            tf_idf=tf_idf,
            target_count=t_count,
            reference_count=r_doc_count,
            target_docs=t_doc_count,
            reference_docs=r_doc_count,
        ))

    # Sort by distinctiveness score, descending
    scores.sort(key=lambda s: s.tf_idf, reverse=True)

    if top_n:
        scores = scores[:top_n]

    return scores


def compare_subsets(
    corpus_docs: list[ExtractedDocument],
    group_key: str,
) -> dict[str, list[ConceptScore]]:
    """
    Compare all groups defined by a tag key against each other.

    For each group value, computes distinctiveness of that group's documents
    against all other documents. Useful for questions like "what concepts
    are unique to each contributor?"

    Args:
        corpus_docs: All documents in the corpus.
        group_key: Tag key to group by (e.g. "contributor").

    Returns:
        Dict mapping group_value -> list of ConceptScore.
    """
    # Partition documents by group
    groups: dict[str, list[ExtractedDocument]] = {}
    for doc in corpus_docs:
        group_val = doc.tags.get(group_key, "_untagged")
        groups.setdefault(group_val, []).append(doc)

    results = {}
    for group_val, group_docs in groups.items():
        # Reference = everything not in this group
        ref_docs = [d for d in corpus_docs if d.doc_id not in
                    {gd.doc_id for gd in group_docs}]
        results[group_val] = distinctiveness(group_docs, ref_docs, top_n=20)

    return results


def temporal_distinctiveness(
    recent: list[ExtractedDocument],
    historical: list[ExtractedDocument],
    top_n: int = 20,
) -> list[ConceptScore]:
    """
    What concepts are new/emerging? Compares recent deposits against
    historical ones.

    This is just a convenience wrapper around distinctiveness() with
    clearer naming for the temporal use case.
    """
    return distinctiveness(recent, historical, top_n=top_n)
