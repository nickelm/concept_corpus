"""
Corpus manager.

Maintains a directory of extracted JSON documents and provides:
- Incremental addition of documents
- Frequency tables over concepts
- Filtered subsets for distinctiveness queries
- Periodic rebuild of global statistics
"""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

from .models import ExtractedDocument, Concept


class Corpus:
    """
    A persistent collection of extracted documents.

    Storage layout:
        corpus_dir/
            docs/
                {doc_id}.json      # one per ingested document
            index.json             # lightweight index: doc_id -> title, source, tags, added_at
    """

    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.docs_dir = self.corpus_dir / "docs"
        self.index_path = self.corpus_dir / "index.json"

        # Ensure directories exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                self._index = json.load(f)
        else:
            self._index = {}

        # In-memory cache of loaded documents (lazy)
        self._docs_cache: dict[str, ExtractedDocument] = {}

    # ── Adding documents ──────────────────────────────────────────────

    def add(self, doc: ExtractedDocument) -> str:
        """
        Add an extracted document to the corpus.
        Persists the JSON immediately. Returns the doc_id.
        """
        # Save document JSON
        doc_path = self.docs_dir / f"{doc.doc_id}.json"
        with open(doc_path, "w") as f:
            f.write(doc.to_json())

        # Update index
        self._index[doc.doc_id] = {
            "title": doc.title,
            "source": doc.source,
            "tags": doc.tags,
            "added_at": doc.added_at,
            "n_concepts": len(doc.concepts),
        }
        self._save_index()

        # Update cache
        self._docs_cache[doc.doc_id] = doc

        return doc.doc_id

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    # ── Retrieving documents ──────────────────────────────────────────

    def get(self, doc_id: str) -> ExtractedDocument:
        """Load a single document by ID."""
        if doc_id in self._docs_cache:
            return self._docs_cache[doc_id]

        doc_path = self.docs_dir / f"{doc_id}.json"
        if not doc_path.exists():
            raise KeyError(f"Document {doc_id} not found")

        with open(doc_path, "r") as f:
            doc = ExtractedDocument.from_json(f.read())

        self._docs_cache[doc_id] = doc
        return doc

    def all_docs(self) -> Iterator[ExtractedDocument]:
        """Iterate over all documents (loads from disk as needed)."""
        for doc_id in self._index:
            yield self.get(doc_id)

    def filter(
        self,
        tag_key: Optional[str] = None,
        tag_value: Optional[str] = None,
        added_after: Optional[str] = None,
        added_before: Optional[str] = None,
    ) -> list[ExtractedDocument]:
        """
        Filter documents by metadata.

        Examples:
            corpus.filter(tag_key="contributor", tag_value="hugo")
            corpus.filter(added_after="2026-03-01")
        """
        results = []
        for doc_id, meta in self._index.items():
            # Tag filter
            if tag_key and tag_value:
                if meta.get("tags", {}).get(tag_key) != tag_value:
                    continue

            # Time filters
            if added_after and meta["added_at"] < added_after:
                continue
            if added_before and meta["added_at"] > added_before:
                continue

            results.append(self.get(doc_id))

        return results

    # ── Corpus statistics ─────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._index)

    def concept_frequency(
        self,
        docs: Optional[list[ExtractedDocument]] = None,
    ) -> Counter:
        """
        Count how many documents contain each concept name.
        This is the "document frequency" (DF) side of the equation.
        """
        if docs is None:
            docs = list(self.all_docs())

        df = Counter()
        for doc in docs:
            # Count each concept once per document
            seen = set()
            for concept in doc.concepts:
                name = concept.name.lower().strip()
                if name not in seen:
                    df[name] += 1
                    seen.add(name)
        return df

    def concept_catalog(
        self,
        docs: Optional[list[ExtractedDocument]] = None,
    ) -> dict[str, list[Concept]]:
        """
        Build a catalog: concept_name -> list of Concept objects across all docs.
        Useful for seeing how the same concept appears in different documents.
        """
        if docs is None:
            docs = list(self.all_docs())

        catalog: dict[str, list[Concept]] = {}
        for doc in docs:
            for concept in doc.concepts:
                name = concept.name.lower().strip()
                catalog.setdefault(name, []).append(concept)
        return catalog

    def list_tags(self, tag_key: str) -> list[str]:
        """List all unique values for a given tag key."""
        values = set()
        for meta in self._index.values():
            v = meta.get("tags", {}).get(tag_key)
            if v:
                values.add(v)
        return sorted(values)

    def summary(self) -> dict:
        """Return a summary of the corpus state."""
        all_concepts = Counter()
        for doc in self.all_docs():
            for c in doc.concepts:
                all_concepts[c.name.lower().strip()] += 1

        return {
            "n_documents": self.size,
            "n_unique_concepts": len(all_concepts),
            "n_total_concept_mentions": sum(all_concepts.values()),
            "top_concepts": all_concepts.most_common(20),
            "maturity": _maturity_label(self.size),
        }


def _maturity_label(n: int) -> str:
    if n < 5:
        return "nascent (< 5 docs — distinctiveness scores unreliable)"
    elif n < 20:
        return "emerging (5-20 docs — pairwise comparisons useful)"
    elif n < 50:
        return "developing (20-50 docs — cluster structure forming)"
    else:
        return "mature (50+ docs — full hierarchy meaningful)"
