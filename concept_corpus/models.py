"""
Data models for the concept corpus.

A Document is the raw input (PDF, URL, text).
An ExtractedDocument is the lightweight JSON representation produced after
one-time LLM extraction — this is what gets persisted and used for all
downstream computation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
import hashlib


@dataclass
class Concept:
    """A single semantic concept extracted from a document."""
    name: str                          # e.g. "physiological data mapping"
    concept_type: str                  # method | theory | dataset | tool | domain | finding | problem
    description: str                   # one-sentence explanation
    related: list[str] = field(default_factory=list)  # related concept names

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.concept_type,
            "description": self.description,
            "related": self.related,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        return cls(
            name=d["name"],
            concept_type=d.get("type", "unknown"),
            description=d.get("description", ""),
            related=d.get("related", []),
        )


@dataclass
class ExtractedDocument:
    """
    The persistent, lightweight representation of a document after LLM extraction.
    This is what gets saved as JSON and loaded for all downstream computation.
    No raw text is stored here — just the extracted concepts and metadata.
    """
    doc_id: str                        # unique identifier
    title: str                         # document title
    source: str                        # filename, URL, or free-text origin
    concepts: list[Concept]            # extracted concept tuples
    summary: str                       # one-paragraph summary from LLM
    tags: dict[str, str] = field(default_factory=dict)  # arbitrary metadata (contributor, type, etc.)
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "concepts": [c.to_dict() for c in self.concepts],
            "summary": self.summary,
            "tags": self.tags,
            "added_at": self.added_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractedDocument":
        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            source=d["source"],
            concepts=[Concept.from_dict(c) for c in d["concepts"]],
            summary=d["summary"],
            tags=d.get("tags", {}),
            added_at=d.get("added_at", datetime.now().isoformat()),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ExtractedDocument":
        return cls.from_dict(json.loads(json_str))


def make_doc_id(source: str, title: str = "") -> str:
    """Generate a stable document ID from source and title."""
    content = f"{source}::{title}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]
