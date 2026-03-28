# SPEC: Viewer Aggregation & Blob Visualization

## Overview

Revise `viewer.html` to support visual aggregation of documents using the hierarchical clustering from `layout.json`. A slider controls the aggregation level, smoothly transitioning from individual document dots to merged cluster blobs. Blobs are rendered using a metaball-style field so that spatially separated cluster members appear as distinct blobs sharing a color rather than one misleading convex hull.

## Data Source

The viewer loads a `layout.json` produced by `concept-corpus layout`. The JSON contains:

- `documents[]`: each with `doc_id`, `title`, `tags`, `n_concepts`, `source`, and `coordinates` (keyed by `pc0`, `pc1`, ..., `pcN`)
- `axes[]`: PCA axes with `index`, `label`, `variance_explained`, `top_concepts[]`
- `dendrogram`: nested binary tree with `node_id`, `left`, `right`, `distance`, `doc_ids[]`, `label`, `size`
- `n_documents`, `n_concepts`

## Aggregation Slider

Add a horizontal slider to the header bar, between the axis selectors and the stats.

- **Slider range:** maps to the dendrogram cut height. Left = minimum distance (every document is its own cluster). Right = maximum distance (all documents in one cluster).
- **Implementation:** use `distance` values from the dendrogram to determine cut levels. At a given slider position, walk the dendrogram and collect all nodes whose `distance` exceeds the threshold as cluster roots. Their `doc_ids` arrays define the cluster memberships.
- **Label on slider:** show the current number of clusters, e.g. "13 clusters" or "4 clusters" or "1 cluster".
- **The slider should feel continuous**, not stepped. The number of clusters changes at discrete thresholds but the slider itself is smooth.

## Cluster Blob Rendering

### At full resolution (slider all the way left)

Render individual dots as the current viewer does. No blobs.

### At any aggregation level (slider moved right)

Each cluster is rendered as a blob using a **metaball-style field approximation**:

1. For each cluster, collect the screen-space positions of all member documents.
2. Each member position emits a radial field: `f(x,y) = R² / ((x - cx)² + (y - cy)²)` where R is a radius parameter (e.g. 40px).
3. Sum the fields from all members of the same cluster.
4. Draw a filled contour where the summed field exceeds a threshold (e.g. 1.0).

**This means:**
- If two member dots are close together, their fields merge into one smooth blob.
- If two member dots are far apart, they appear as two separate blobs.
- The blobs share the same fill color (semi-transparent) so the viewer can see they belong to the same cluster even when spatially separated.

**Performance notes:**
- Evaluate the field on a coarse grid (e.g. 200×150 pixels) and use marching squares to extract contours, then draw those contours on the canvas.
- Alternatively, if marching squares is too complex: approximate with overlapping semi-transparent circles at each member position with large radii (e.g. 50px). This gives a similar visual effect with much simpler code. Circles from the same cluster share the same color and alpha, so overlapping circles merge visually. **This simpler approach is acceptable for a prototype.**

### Cluster colors

Use a palette of 8-10 distinct muted colors for cluster fills. These are independent of contributor colors. Use low alpha (0.15-0.25) for the blob fill and slightly higher alpha (0.4) for the blob border/edge.

### Member dots inside blobs

When a cluster is aggregated into a blob, still draw small dots (radius 3-4px) at member positions inside the blob, colored by contributor. This preserves the contributor information within the cluster view.

### Blob size indication

The blob should visually communicate cluster size. For the simple circle approach, scale the circle radius by `sqrt(cluster_size)` relative to a base radius. For metaballs, the natural field summation already produces larger blobs for more members.

## Label Behavior

### Default state (no interaction)

- **No labels shown.** The view is clean — just blobs/dots and axis labels.

### On hover over a blob

- Show the **cluster label** (from the dendrogram node's `label` field) near the blob centroid.
- Show the **member count** (e.g. "4 documents").
- Optionally reveal individual document titles as small text near each member dot inside the blob.

### On hover over an individual dot (at full resolution)

- Show document title and metadata in the tooltip, as current viewer does.

### On sidebar cluster highlight

- Show the cluster label at the blob centroid on the canvas.
- Dim all other blobs/dots.
- Draw connecting lines between members of the highlighted cluster (as current viewer does).

## Sidebar Updates

### Cluster list

The sidebar cluster list should update dynamically when the slider moves. At each aggregation level, show the current clusters with their labels and sizes. The dendrogram depth-based approach in the current viewer should be replaced with the slider-driven cut.

### Document list

Keep the document list unchanged. Hovering a document in the sidebar should highlight its dot and the blob it belongs to at the current aggregation level.

## Transition Behavior

When the slider changes:
- Recalculate cluster memberships from the dendrogram cut.
- Re-render blobs and dots.
- No animation needed for prototype — instant re-render is fine.
- Update the sidebar cluster list to reflect new groupings.

## Existing Features to Preserve

- PCA axis selection dropdowns (X and Y) — these continue to control which two axes determine the 2D positions.
- Contributor color legend in sidebar.
- File drop / load area for `layout.json`.
- Stats display (doc count, concept count).
- Tooltip on hover.
- Resize handling.
- Dark theme styling.

## Technical Constraints

- Single HTML file, no external dependencies except Google Fonts (already used).
- Pure canvas rendering (no SVG, no D3).
- Must handle up to ~100 documents without performance issues.
- All rendering logic in the `render()` function, triggered by slider change, axis change, hover, or resize.

## File

Edit `viewer.html` in the project root. It is a single self-contained HTML file.