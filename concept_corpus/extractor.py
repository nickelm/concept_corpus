"""
Concept extraction via LLM.

Takes raw text and produces an ExtractedDocument with structured concept tuples.
Uses LiteLLM as the backend so any model provider works.
"""

import json
import re
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .models import Concept, ExtractedDocument, make_doc_id


EXTRACTION_PROMPT = """You are a research analyst building a shared concept index across a corpus of academic papers. Extract the key intellectual concepts from the following document.

CRITICAL NAMING RULES:
- Use established, canonical terminology from the field. Prefer "information visualization" over "interactive visual data exploration techniques."
- Use the shortest standard name: "augmented reality" not "augmented reality technology," "topic modeling" not "statistical topic modeling approaches."
- If a concept has a widely known name in the literature, use that exact name.
- Use lowercase unless it is a proper noun or acronym.
- Aim for 2-4 word concept names.

For each concept, provide:
- name: canonical label (2-4 words, using standard terminology)
- type: one of [method, theory, dataset, tool, domain, finding, problem, framework, technique, application]
- description: one sentence explaining the concept as used in this document
- related: list of 0-3 related concept names (using the same canonical naming rules)

Also provide:
- title: the document's title (infer if not explicit)
- summary: a one-paragraph summary of the document's main contribution or content

Respond with ONLY valid JSON in this exact format:
{
  "title": "...",
  "summary": "...",
  "concepts": [
    {
      "name": "...",
      "type": "...",
      "description": "...",
      "related": ["..."]
    }
  ]
}

Extract between 8 and 15 concepts. Focus on the core intellectual contributions:
- The main methods, techniques, or frameworks introduced or used
- The key domains and application areas
- Central theories or design principles invoked
- Important tools, datasets, or systems referenced

Skip: generic concepts that apply to any paper (e.g. "literature review," "user study," "research methodology"), bibliographic metadata, author names, section headings.

DOCUMENT:
"""


def extract_concepts(
    text: str,
    source: str = "",
    tags: Optional[dict[str, str]] = None,
    model: str = "anthropic/claude-haiku-4-5-20251001",
    api_base: Optional[str] = None,
) -> ExtractedDocument:
    """
    Extract concepts from raw text using an LLM.

    Args:
        text: The document text to analyze.
        source: Origin of the document (filename, URL, etc.).
        tags: Arbitrary metadata to attach (contributor, type, etc.).
        model: LiteLLM model string.
        api_base: Optional API base URL (for LiteLLM proxy).

    Returns:
        ExtractedDocument with extracted concepts and metadata.
    """
    import litellm

    # Truncate very long documents to stay within context limits
    max_chars = 80_000  # ~20k tokens, safe for most models
    if len(text) > max_chars:
        # Keep beginning and end
        half = max_chars // 2
        text = text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]

    kwargs = {
        "model": model,
        "messages": [
            {"role": "user", "content": EXTRACTION_PROMPT + text}
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.completion(**kwargs)
    raw = response.choices[0].message.content

    # Parse JSON from response (handle markdown code blocks)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    data = json.loads(raw)

    title = data.get("title", source or "Untitled")
    doc_id = make_doc_id(source, title)

    return ExtractedDocument(
        doc_id=doc_id,
        title=title,
        source=source,
        concepts=[Concept.from_dict(c) for c in data.get("concepts", [])],
        summary=data.get("summary", ""),
        tags=tags or {},
    )


def extract_from_file(
    filepath: str,
    tags: Optional[dict[str, str]] = None,
    model: str = "anthropic/claude-haiku-4-5-20251001",
    api_base: Optional[str] = None,
) -> ExtractedDocument:
    """Extract concepts from a text or PDF file."""
    import os

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        text = _read_pdf(filepath)
    elif ext in (".txt", ".md", ".tex", ".bib"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    return extract_concepts(
        text=text,
        source=filepath,
        tags=tags,
        model=model,
        api_base=api_base,
    )


def _read_pdf(filepath: str) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n\n".join(pages)
    except ImportError:
        # Fallback: try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "PDF reading requires PyMuPDF (pip install pymupdf) "
                "or pdfplumber (pip install pdfplumber)"
            )