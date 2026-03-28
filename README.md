# concept-corpus

Semantic TF-IDF for LLM-extracted concepts. Build a corpus incrementally, extract structured concepts from documents using an LLM, and compute distinctiveness scores between arbitrary subsets.

## The idea

Classic TF-IDF finds words that are common in one document but rare across the corpus. This library does the same thing, but at the **concept level** — using an LLM to extract structured idea representations, then computing distinctiveness over those representations.

This enables questions like:
- What concepts in this paper are unlike anything else in my collection?
- What themes distinguish contributor A's reading from contributor B's?
- What new ideas appeared in the last month that weren't present before?

## Installation

```bash
pip install -e .                    # basic install
pip install -e ".[pdf]"             # with PDF support (PyMuPDF)
pip install -e ".[all]"             # everything
```

Requires an LLM API key. Set `ANTHROPIC_API_KEY` for Claude models, or use `--api-base` to point at a LiteLLM proxy.

## Quick start (CLI)

```bash
python -m concept_corpus --corpus ./test_corpus ingest <file> --tag contributor=niklas
```

```bash
# Ingest a PDF, tagging it with a contributor
concept-corpus ingest paper.pdf --tag contributor=hugo

# Ingest raw text (e.g. from a note or idea)
concept-corpus ingest-text "AR visualization of physiological data using density-based clustering" \
    --tag contributor=niklas --tag type=idea

# Check corpus status
concept-corpus status

# Score one document's distinctiveness against the rest
concept-corpus score abc123def456

# Compare contributors — what's unique to each person's deposits?
concept-corpus compare --group-by contributor

# What's new this month?
concept-corpus recent --since 2026-03-01
```

### Using a LiteLLM proxy

```bash
concept-corpus ingest paper.pdf --model gpt-4o-mini --api-base http://localhost:4000
```

## Quick start (Python)

```python
from concept_corpus import Corpus, extract_concepts, distinctiveness

# Create or open a corpus
corpus = Corpus("./my_corpus")

# Extract concepts from text (calls LLM once)
doc = extract_concepts(
    "This paper presents a method for visualizing physiological data...",
    source="paper.pdf",
    tags={"contributor": "hugo", "type": "paper"},
)

# Add to corpus (persists JSON immediately)
corpus.add(doc)

# Later: score distinctiveness against the whole corpus
all_docs = list(corpus.all_docs())
ref = [d for d in all_docs if d.doc_id != doc.doc_id]
scores = distinctiveness(target=[doc], reference=ref, top_n=10)

for s in scores:
    print(f"{s.tf_idf:.3f}  {s.concept_name}")
```

## Architecture

```
Document (PDF, URL, text)
    │
    ▼  [LLM extraction — expensive, once per document]
ExtractedDocument (JSON)
    │   - title, summary
    │   - concepts: [{name, type, description, related}]
    │   - tags: {contributor, type, ...}
    │
    ▼  [stored to disk as JSON — lightweight]
Corpus
    │   - docs/{doc_id}.json
    │   - index.json
    │
    ▼  [scoring — cheap, runs on JSON only]
Distinctiveness scores
    - target subset vs. reference subset
    - concept-level TF-IDF
```

Key design decisions:
- **Each document is extracted once**, producing a small JSON file (typically 2-5 KB).
- **All downstream computation uses only the JSON files**, never the original documents.
- **Periodic rebuilds** just reload JSON — no LLM calls, no PDF parsing.
- **Distinctiveness is computed between arbitrary subsets**, not hardcoded to any particular grouping.

## Storage layout

```
my_corpus/
  docs/
    a1b2c3d4e5f6.json    # one per ingested document
    ...
  index.json              # lightweight lookup: doc_id → title, tags, etc.
```

## Concept types

The extractor classifies concepts into: `method`, `theory`, `dataset`, `tool`, `domain`, `finding`, `problem`, `framework`, `technique`, `application`.

## License

MIT
