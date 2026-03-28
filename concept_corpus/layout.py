"""
Layout computation for corpus visualization.

Produces:
- A document-concept matrix from stored JSONs
- 2D coordinates via PCA (deterministic)
- A hierarchical clustering dendrogram (deterministic)
- Cluster labels derived from top distinctive concepts (no LLM needed)
- Named PCA axes based on top-loading concepts

All outputs are JSON-serializable for frontend consumption.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .models import ExtractedDocument
from .scorer import distinctiveness


# ── Document-Concept Matrix ──────────────────────────────────────────

def build_concept_matrix(
    docs: list[ExtractedDocument],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a documents × concepts matrix from extracted documents.

    Returns:
        matrix: numpy array of shape (n_docs, n_concepts), values are
                concept frequency within each document (0 or 1+ occurrences).
        doc_ids: list of document IDs (row labels).
        concept_names: list of concept names (column labels).
    """
    # Collect all unique concept names
    all_concepts: set[str] = set()
    for doc in docs:
        for c in doc.concepts:
            all_concepts.add(c.name.lower().strip())

    concept_names = sorted(all_concepts)
    concept_index = {name: i for i, name in enumerate(concept_names)}
    doc_ids = [doc.doc_id for doc in docs]

    matrix = np.zeros((len(docs), len(concept_names)), dtype=np.float64)

    for i, doc in enumerate(docs):
        for c in doc.concepts:
            name = c.name.lower().strip()
            j = concept_index[name]
            matrix[i, j] += 1.0

    return matrix, doc_ids, concept_names


# ── PCA Projection ───────────────────────────────────────────────────

@dataclass
class PCAxis:
    """A principal component axis with its top-loading concepts."""
    index: int
    variance_explained: float
    top_concepts: list[tuple[str, float]]  # (concept_name, loading)

    def label(self) -> str:
        """Mechanical label from top 3 concepts."""
        names = [name for name, _ in self.top_concepts[:3]]
        return " / ".join(names)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "variance_explained": round(self.variance_explained, 4),
            "top_concepts": [
                {"name": n, "loading": round(l, 4)}
                for n, l in self.top_concepts
            ],
            "label": self.label(),
        }


def compute_pca(
    matrix: np.ndarray,
    concept_names: list[str],
    n_components: int = 10,
    top_concepts_per_axis: int = 5,
) -> tuple[np.ndarray, list[PCAxis]]:
    """
    Compute PCA on the document-concept matrix.

    Returns:
        coords: array of shape (n_docs, n_components) — document coordinates.
        axes: list of PCAxis with labeled components.
    """
    # Normalize rows (L2) so document length doesn't dominate
    normed = normalize(matrix, norm="l2", axis=1)

    # Cap components at matrix rank
    max_components = min(n_components, *matrix.shape)
    pca = PCA(n_components=max_components)
    coords = pca.fit_transform(normed)

    axes = []
    for i in range(max_components):
        loadings = pca.components_[i]
        # Get top concepts by absolute loading
        top_indices = np.argsort(np.abs(loadings))[::-1][:top_concepts_per_axis]
        top = [(concept_names[j], float(loadings[j])) for j in top_indices]

        axes.append(PCAxis(
            index=i,
            variance_explained=float(pca.explained_variance_ratio_[i]),
            top_concepts=top,
        ))

    return coords, axes


# ── Hierarchical Clustering ──────────────────────────────────────────

@dataclass
class ClusterNode:
    """A node in the cluster dendrogram."""
    node_id: int
    left: Optional["ClusterNode"] = None
    right: Optional["ClusterNode"] = None
    distance: float = 0.0
    doc_ids: list[str] = field(default_factory=list)  # leaf doc_ids under this node
    label: str = ""                                    # auto-generated label
    size: int = 1

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id,
            "distance": round(self.distance, 4),
            "doc_ids": self.doc_ids,
            "label": self.label,
            "size": self.size,
        }
        if self.left:
            d["left"] = self.left.to_dict()
        if self.right:
            d["right"] = self.right.to_dict()
        return d


def compute_hierarchy(
    docs: list[ExtractedDocument],
    matrix: np.ndarray,
    doc_ids: list[str],
    method: str = "ward",
) -> ClusterNode:
    """
    Compute agglomerative hierarchical clustering.

    Returns the root of a binary dendrogram tree.
    Each internal node is labeled with its top distinctive concepts
    (compared to its sibling subtree).

    Args:
        docs: The extracted documents.
        matrix: Document-concept matrix.
        doc_ids: Row labels for the matrix.
        method: Linkage method (ward, complete, average, single).
    """
    if len(docs) < 2:
        if docs:
            return ClusterNode(
                node_id=0,
                doc_ids=[docs[0].doc_id],
                label=docs[0].title,
                size=1,
            )
        return ClusterNode(node_id=0)

    # Normalize before computing distances
    normed = normalize(matrix, norm="l2", axis=1)

    # Compute linkage
    Z = linkage(normed, method=method, metric="euclidean")

    # Build doc lookup
    doc_by_id = {d.doc_id: d for d in docs}

    # Convert scipy tree to our ClusterNode tree
    scipy_root = to_tree(Z, rd=True)[0]

    def _build_node(scipy_node) -> ClusterNode:
        if scipy_node.is_leaf():
            did = doc_ids[scipy_node.id]
            doc = doc_by_id[did]
            return ClusterNode(
                node_id=scipy_node.id,
                doc_ids=[did],
                label=doc.title[:60],
                size=1,
                distance=0.0,
            )

        left = _build_node(scipy_node.get_left())
        right = _build_node(scipy_node.get_right())

        merged_ids = left.doc_ids + right.doc_ids
        node = ClusterNode(
            node_id=scipy_node.id,
            left=left,
            right=right,
            doc_ids=merged_ids,
            distance=float(scipy_node.dist),
            size=left.size + right.size,
        )

        # Label: distinctive concepts of this cluster vs its complement
        # We'll label in a separate pass (needs full doc list)
        return node

    root = _build_node(scipy_root)

    # Label internal nodes using distinctiveness
    _label_nodes(root, docs)

    return root


def _label_nodes(node: ClusterNode, all_docs: list[ExtractedDocument]):
    """
    Recursively label internal nodes with their top distinctive concepts.
    Each node is labeled by what distinguishes it from the rest of the corpus.
    """
    if node.is_leaf():
        return

    # Label children first
    _label_nodes(node.left, all_docs)
    _label_nodes(node.right, all_docs)

    # Get docs in this subtree vs everything else
    subtree_ids = set(node.doc_ids)
    doc_by_id = {d.doc_id: d for d in all_docs}

    target = [doc_by_id[did] for did in node.doc_ids if did in doc_by_id]
    reference = [d for d in all_docs if d.doc_id not in subtree_ids]

    if reference and target:
        scores = distinctiveness(target, reference, top_n=3)
        if scores:
            node.label = " / ".join(s.concept_name for s in scores[:3])
        else:
            node.label = f"cluster ({node.size} docs)"
    else:
        # Root node or no reference — label with most common concepts
        concept_counts = Counter()
        for doc in target:
            for c in doc.concepts:
                concept_counts[c.name.lower().strip()] += 1
        top = concept_counts.most_common(3)
        node.label = " / ".join(name for name, _ in top) if top else f"corpus ({node.size} docs)"


# ── Flat Cluster Cuts ─────────────────────────────────────────────────

def cut_clusters(
    docs: list[ExtractedDocument],
    matrix: np.ndarray,
    doc_ids: list[str],
    n_clusters: int = 5,
    method: str = "ward",
) -> dict[int, list[str]]:
    """
    Cut the dendrogram at a level producing n_clusters groups.

    Returns:
        Dict mapping cluster_id -> list of doc_ids.
    """
    normed = normalize(matrix, norm="l2", axis=1)
    Z = linkage(normed, method=method, metric="euclidean")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters: dict[int, list[str]] = {}
    for i, cluster_id in enumerate(labels):
        clusters.setdefault(int(cluster_id), []).append(doc_ids[i])

    return clusters


# ── Full Layout Computation ──────────────────────────────────────────

def compute_layout(
    docs: list[ExtractedDocument],
    n_pca_components: int = 10,
    linkage_method: str = "ward",
) -> dict:
    """
    Compute the full layout for a set of documents.

    Returns a JSON-serializable dict containing:
    - documents: list of {doc_id, title, tags, x, y, ...} with coordinates
                 for each PCA axis pair
    - axes: list of PCA axes with labels and top concepts
    - dendrogram: nested cluster tree with labels
    - n_documents: corpus size

    The frontend can pick any two axes for the 2D projection and use
    the dendrogram for coloring / grouping.
    """
    if not docs:
        return {"documents": [], "axes": [], "dendrogram": None, "n_documents": 0}

    # Build concept matrix
    matrix, doc_ids, concept_names = build_concept_matrix(docs)

    # PCA
    coords, axes = compute_pca(matrix, concept_names, n_components=n_pca_components)

    # Hierarchy
    dendrogram = compute_hierarchy(docs, matrix, doc_ids, method=linkage_method)

    # Assemble per-document records
    doc_records = []
    for i, doc in enumerate(docs):
        record = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source": doc.source,
            "tags": doc.tags,
            "n_concepts": len(doc.concepts),
            "coordinates": {
                f"pc{j}": round(float(coords[i, j]), 4)
                for j in range(coords.shape[1])
            },
        }
        doc_records.append(record)

    return {
        "documents": doc_records,
        "axes": [a.to_dict() for a in axes],
        "dendrogram": dendrogram.to_dict(),
        "n_documents": len(docs),
        "n_concepts": len(concept_names),
    }
