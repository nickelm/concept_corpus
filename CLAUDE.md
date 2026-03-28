# concept-corpus

Semantic TF-IDF for LLM-extracted concepts. Part of the Tessera/Knowledge Funnels project.

## What this is

A Python library that extracts structured concepts from documents using an LLM, builds a persistent corpus of lightweight JSON representations, and computes distinctiveness scores between arbitrary subsets. Think TF-IDF, but over semantic concepts instead of raw tokens.

## Architecture

```
Document (PDF, URL, text)
  → extractor.py (LLM call, once per document)
  → ExtractedDocument JSON (persisted to disk, ~2KB each)
  → corpus.py (manages storage, filtering, frequency tables)
  → scorer.py (concept-level TF-IDF between target/reference subsets)
  → layout.py (PCA coordinates, hierarchical clustering, cluster labels)
```

Key principle: each document is extracted once at ingest time. All downstream computation (scoring, layout, clustering) operates only on the stored JSON files — never re-reads original documents or makes additional LLM calls.

## Module overview

- `models.py` — Data classes: `Concept`, `ExtractedDocument`, `make_doc_id`
- `extractor.py` — LLM-based concept extraction via LiteLLM. Supports PDF (via PyMuPDF) and plain text.
- `corpus.py` — `Corpus` class: persistent storage in `docs/{doc_id}.json` + `index.json`. Supports tag-based and temporal filtering.
- `scorer.py` — `distinctiveness(target, reference)` computes concept-level TF-IDF. `compare_subsets` partitions by any tag. `temporal_distinctiveness` compares recent vs historical.
- `layout.py` — `compute_layout(docs)` returns JSON with PCA coordinates, labeled axes, and a hierarchical dendrogram with auto-generated cluster labels.
- `__main__.py` — CLI interface.

## CLI usage

Arguments before subcommand, e.g.:
```bash
python -m concept_corpus --corpus ./my_corpus status
python -m concept_corpus --corpus ./my_corpus ingest paper.pdf --tag contributor=hugo
python -m concept_corpus --corpus ./my_corpus score DOC_ID
python -m concept_corpus --corpus ./my_corpus layout -o layout.json
python -m concept_corpus --corpus ./my_corpus compare --group-by contributor
python -m concept_corpus --corpus ./my_corpus --api-base http://localhost:4000 --model gpt-4o-mini ingest paper.pdf
```

## Design decisions

- **Distinctiveness is computed between arbitrary subsets**, not hardcoded to contributors or time. The `target` and `reference` are parameters. Tags (contributor, type, etc.) are metadata on documents, not part of the scoring logic.
- **Hierarchy is fixed, projection is a view.** The dendrogram comes from agglomerative clustering on concept vectors and doesn't change when the user picks different PCA axes. Cluster labels are mechanical (top distinctive concepts), not LLM-generated.
- **Layout is deterministic.** PCA + Ward linkage = same input always produces same output. No random seeds, no stochastic embeddings.
- **Concept matching is exact string match** on lowercased concept names. A future improvement would be embedding-based fuzzy matching.

## Storage layout

```
corpus_dir/
  docs/
    {doc_id}.json    # one per ingested document
  index.json         # doc_id → title, source, tags, added_at, n_concepts
```

## Dependencies

- `litellm` — LLM backend for extraction (supports Anthropic, OpenAI, local models, LiteLLM proxy)
- `numpy`, `scipy`, `scikit-learn` — matrix operations, PCA, hierarchical clustering
- `pymupdf` (optional) — PDF text extraction

## Environment

- Python 3.12+ (installed from python.org, not Windows Store)
- Uses venv: `.venv/` in project root
- Set `ANTHROPIC_API_KEY` or use `--api-base` for LiteLLM proxy

## Context: Tessera / Knowledge Funnels

This library is the backend for the Knowledge Funnels system's extraction and generation layers. Knowledge Funnels captures a research group's collective knowledge through multiple input channels, extracts structured representations, and applies idea operations to generate novel research pitches. The layout output feeds a "knowledge map" visualization in the frontend. The distinctiveness scores identify what makes each contributor's intellectual landscape unique relative to the group.