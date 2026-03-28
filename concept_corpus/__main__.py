"""
Command-line interface for concept_corpus.

Usage:
    # Ingest a document (extracts concepts via LLM, saves JSON)
    python -m concept_corpus ingest paper.pdf --corpus ./my_corpus --tag contributor=hugo

    # Ingest from raw text
    python -m concept_corpus ingest-text "Some interesting idea..." --corpus ./my_corpus

    # Show corpus summary
    python -m concept_corpus status --corpus ./my_corpus

    # Compute distinctiveness of a document against the rest
    python -m concept_corpus score DOC_ID --corpus ./my_corpus

    # Compare all contributors
    python -m concept_corpus compare --corpus ./my_corpus --group-by contributor

    # Show what's new recently
    python -m concept_corpus recent --corpus ./my_corpus --since 2026-03-01
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="concept_corpus",
        description="Semantic TF-IDF for LLM-extracted concepts",
    )
    parser.add_argument("--corpus", default="./corpus", help="Corpus directory")
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5-20251001",
                        help="LiteLLM model string for extraction")
    parser.add_argument("--api-base", default=None,
                        help="API base URL (for LiteLLM proxy)")

    sub = parser.add_subparsers(dest="command")

    # ── ingest ────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest a file (PDF, txt, md, etc.)")
    p_ingest.add_argument("file", help="Path to file")
    p_ingest.add_argument("--tag", action="append", default=[],
                          help="Tag as key=value (repeatable)")

    # ── ingest-text ───────────────────────────────────────────────────
    p_text = sub.add_parser("ingest-text", help="Ingest raw text")
    p_text.add_argument("text", help="Text to ingest")
    p_text.add_argument("--source", default="cli-input", help="Source label")
    p_text.add_argument("--tag", action="append", default=[],
                        help="Tag as key=value (repeatable)")

    # ── status ────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show corpus summary")

    # ── score ─────────────────────────────────────────────────────────
    p_score = sub.add_parser("score", help="Score a document's distinctiveness")
    p_score.add_argument("doc_id", help="Document ID to score")
    p_score.add_argument("--top", type=int, default=15, help="Top N concepts")

    # ── compare ───────────────────────────────────────────────────────
    p_compare = sub.add_parser("compare", help="Compare groups")
    p_compare.add_argument("--group-by", required=True,
                           help="Tag key to group by (e.g. contributor)")

    # ── recent ────────────────────────────────────────────────────────
    p_recent = sub.add_parser("recent", help="Show distinctive recent additions")
    p_recent.add_argument("--since", required=True,
                          help="ISO date string (e.g. 2026-03-01)")
    p_recent.add_argument("--top", type=int, default=15, help="Top N concepts")

    # ── list ──────────────────────────────────────────────────────────
    sub.add_parser("list", help="List all documents in the corpus")

    # ── layout ────────────────────────────────────────────────────────
    p_layout = sub.add_parser("layout", help="Compute layout JSON for visualization")
    p_layout.add_argument("--output", "-o", default=None,
                          help="Output JSON file (default: stdout)")
    p_layout.add_argument("--axes", type=int, default=10,
                          help="Number of PCA axes to compute")
    p_layout.add_argument("--linkage", default="ward",
                          choices=["ward", "complete", "average", "single"],
                          help="Hierarchical clustering linkage method")

    # ── merge ─────────────────────────────────────────────────────────
    p_merge = sub.add_parser("merge", help="Merge synonym concepts across corpus")
    p_merge.add_argument("--mode", choices=["fast", "llm"], default="fast",
                         help="fast = string similarity (free), llm = LLM grouping (one API call)")
    p_merge.add_argument("--threshold", type=float, default=0.6,
                         help="Similarity threshold for fast mode (0-1)")
    p_merge.add_argument("--dry-run", action="store_true",
                         help="Show proposed merges without applying")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    from concept_corpus import (
        Corpus, extract_from_file, extract_concepts,
        distinctiveness, compare_subsets, temporal_distinctiveness,
        compute_layout, merge_by_similarity, merge_by_llm, apply_merges,
    )
    from concept_corpus.merger import deduplicate_within_doc

    corpus = Corpus(args.corpus)

    # ── Dispatch ──────────────────────────────────────────────────────

    if args.command == "ingest":
        tags = _parse_tags(args.tag)
        print(f"Extracting concepts from {args.file}...")
        doc = extract_from_file(
            args.file, tags=tags, model=args.model, api_base=args.api_base,
        )
        doc_id = corpus.add(doc)
        print(f"Added: {doc.title}")
        print(f"  ID: {doc_id}")
        print(f"  Concepts: {len(doc.concepts)}")
        for c in doc.concepts:
            print(f"    [{c.concept_type}] {c.name}")

    elif args.command == "ingest-text":
        tags = _parse_tags(args.tag)
        print("Extracting concepts...")
        doc = extract_concepts(
            args.text, source=args.source, tags=tags,
            model=args.model, api_base=args.api_base,
        )
        doc_id = corpus.add(doc)
        print(f"Added: {doc.title}")
        print(f"  ID: {doc_id}")
        print(f"  Concepts: {len(doc.concepts)}")
        for c in doc.concepts:
            print(f"    [{c.concept_type}] {c.name}")

    elif args.command == "status":
        s = corpus.summary()
        print(f"Corpus: {args.corpus}")
        print(f"  Documents: {s['n_documents']}")
        print(f"  Unique concepts: {s['n_unique_concepts']}")
        print(f"  Total mentions: {s['n_total_concept_mentions']}")
        print(f"  Maturity: {s['maturity']}")
        if s['top_concepts']:
            print(f"  Top concepts:")
            for name, count in s['top_concepts'][:10]:
                print(f"    {name}: {count}")

    elif args.command == "score":
        doc = corpus.get(args.doc_id)
        all_docs = list(corpus.all_docs())
        ref = [d for d in all_docs if d.doc_id != args.doc_id]

        if not ref:
            print("Need at least 2 documents to compute distinctiveness.")
            return

        scores = distinctiveness([doc], ref, top_n=args.top)
        print(f"Distinctive concepts in: {doc.title}")
        print(f"  (scored against {len(ref)} other documents)\n")
        for s in scores:
            bar = "█" * int(s.tf_idf * 20)
            print(f"  {s.tf_idf:6.3f} {bar:20s} {s.concept_name}")
            print(f"         in target: {s.target_docs}, in reference: {s.reference_docs}")

    elif args.command == "compare":
        all_docs = list(corpus.all_docs())
        results = compare_subsets(all_docs, args.group_by)
        for group, scores in results.items():
            print(f"\n{'='*60}")
            print(f"  {args.group_by}: {group}")
            print(f"{'='*60}")
            for s in scores[:10]:
                print(f"  {s.tf_idf:6.3f}  {s.concept_name}")

    elif args.command == "recent":
        recent_docs = corpus.filter(added_after=args.since)
        historical = corpus.filter(added_before=args.since)

        if not recent_docs:
            print(f"No documents added after {args.since}")
            return
        if not historical:
            print("No historical documents to compare against.")
            return

        scores = temporal_distinctiveness(recent_docs, historical, top_n=args.top)
        print(f"Distinctive new concepts (since {args.since}):")
        print(f"  {len(recent_docs)} recent vs {len(historical)} historical docs\n")
        for s in scores:
            print(f"  {s.tf_idf:6.3f}  {s.concept_name}")

    elif args.command == "list":
        for doc_id, meta in corpus._index.items():
            tags_str = ", ".join(f"{k}={v}" for k, v in meta.get("tags", {}).items())
            print(f"  {doc_id}  {meta['title'][:50]:50s}  [{meta['n_concepts']} concepts]  {tags_str}")

    elif args.command == "layout":
        all_docs = list(corpus.all_docs())
        if len(all_docs) < 2:
            print(f"Need at least 2 documents for layout (have {len(all_docs)}).")
            return

        print(f"Computing layout for {len(all_docs)} documents...", file=sys.stderr)
        layout = compute_layout(
            all_docs,
            n_pca_components=args.axes,
            linkage_method=args.linkage,
        )

        output_json = json.dumps(layout, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"Layout written to {args.output}", file=sys.stderr)
            print(f"  {layout['n_documents']} documents, {layout['n_concepts']} concepts", file=sys.stderr)
            print(f"  {len(layout['axes'])} PCA axes", file=sys.stderr)
        else:
            print(output_json)

    elif args.command == "merge":
        all_docs = list(corpus.all_docs())
        if not all_docs:
            print("Corpus is empty.")
            return

        # Count concepts before
        from collections import Counter
        before = Counter()
        for doc in all_docs:
            for c in doc.concepts:
                before[c.name.lower().strip()] += 1
        print(f"Before merge: {len(before)} unique concepts")

        # Compute merge map
        if args.mode == "llm":
            print("Calling LLM to group synonyms...")
            merge_map = merge_by_llm(all_docs, model=args.model, api_base=args.api_base)
        else:
            merge_map = merge_by_similarity(all_docs, threshold=args.threshold)

        # Show proposed merges
        merges_found = {}
        for original, canonical in merge_map.items():
            if original != canonical:
                merges_found.setdefault(canonical, []).append(original)

        if not merges_found:
            print("No synonyms found to merge.")
            return

        print(f"\nProposed merges ({len(merges_found)} groups):")
        for canonical, members in sorted(merges_found.items()):
            print(f"  {canonical}")
            for m in members:
                print(f"    ← {m}")

        if args.dry_run:
            print("\n(dry run — no changes applied)")
            return

        # Apply merges and save
        apply_merges(all_docs, merge_map)
        for doc in all_docs:
            deduplicate_within_doc(doc)
            corpus.add(doc)  # overwrites existing JSON

        # Count after
        after = Counter()
        for doc in all_docs:
            for c in doc.concepts:
                after[c.name.lower().strip()] += 1
        print(f"\nAfter merge: {len(after)} unique concepts (was {len(before)})")


def _parse_tags(tag_args: list[str]) -> dict[str, str]:
    tags = {}
    for t in tag_args:
        if "=" in t:
            k, v = t.split("=", 1)
            tags[k.strip()] = v.strip()
    return tags


if __name__ == "__main__":
    main()