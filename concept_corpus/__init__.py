"""
concept_corpus — Semantic TF-IDF for LLM-extracted concepts.

Build a corpus incrementally, extract structured concepts from documents,
and compute distinctiveness scores between arbitrary subsets.

Quick start:
    from concept_corpus import Corpus, extract_concepts, distinctiveness

    corpus = Corpus("./my_corpus")
    doc = extract_concepts("Some text...", source="paper.pdf")
    corpus.add(doc)
    scores = distinctiveness(target=[doc], reference=list(corpus.all_docs()))
"""

from .models import Concept, ExtractedDocument, make_doc_id
from .corpus import Corpus
from .extractor import extract_concepts, extract_from_file
from .scorer import (
    distinctiveness,
    compare_subsets,
    temporal_distinctiveness,
    ConceptScore,
)
from .layout import compute_layout, build_concept_matrix, compute_hierarchy
from .merger import merge_by_similarity, merge_by_llm, apply_merges

__version__ = "0.1.0"

__all__ = [
    "Concept",
    "ExtractedDocument",
    "make_doc_id",
    "Corpus",
    "extract_concepts",
    "extract_from_file",
    "distinctiveness",
    "compare_subsets",
    "temporal_distinctiveness",
    "ConceptScore",
    "compute_layout",
    "build_concept_matrix",
    "compute_hierarchy",
    "merge_by_similarity",
    "merge_by_llm",
    "apply_merges",
]
