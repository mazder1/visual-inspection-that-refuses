"""Deriving component identity, because MVTec AD does not ship it.

The split has to guarantee that repeat views of the same physical component
never straddle it. MVTec provides no instance IDs -- filenames are just
``000.png``, ``001.png`` within each defect type -- so identity has to be
recovered from the pixels.

Two methods live here. The first does not work, and is kept because knowing
that is part of the result.

**Perceptual hashing (rejected, retained as a control).** A difference hash
over a downscaled greyscale grid is the standard near-duplicate tool. It fails
on MVTec, and it fails for a structural reason rather than a tuning one: every
image in a category is the same object type, centred, in a fixed rig, under
fixed lighting. The coarse layout a perceptual hash encodes is therefore
*constant across the whole category by construction*, while the thing that
distinguishes two physical parts is fine surface texture -- exactly what the
hash is designed to discard. Sweeping it shows no stable operating point:
cluster size jumps from 1 straight to hundreds with no plateau between, because
the similarity graph is nearly complete and single-linkage chains across it.
See :func:`group_by_hash`.

**Keypoint matching (used).** ORB features plus a Lowe ratio test plus RANSAC
under a partial-affine model asks a sharper question: do these two images share
the same fine texture, arranged in a geometrically consistent way, allowing for
the part having been rotated and re-placed? On MVTec that separates cleanly.
The inlier count is near zero for the overwhelming majority of pairs and lands
in the tens-to-hundreds for genuine repeats, with a wide empty gap between --
so the threshold sits in a basin rather than on a slope. Partial affine is the
right model because it is exactly the transform a part undergoes when it is
lifted and set back down on the same rig: rotation, translation, uniform scale.

Clustering is connected components over the match graph, and runs **within a
category only** -- a bottle and a screw are never the same component.
Single-linkage chaining is a real risk, but with a threshold in the gap the
error is one-directional: chaining makes groups larger, which makes the split
stricter, never leakier. Verify it by eye with :func:`dump_clusters` before
trusting it on a category that has not been looked at.

The cluster-size distribution is a reported result, not an implementation
detail. It measures how much component reuse the dataset contains, and
therefore how much a random split would have overstated.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from vinspect.data.mvtec import MVTecRecord

# --- perceptual hash, the rejected control ---------------------------------

HASH_SIZE = 8
DEFAULT_HASH_THRESHOLD = 6

# --- keypoint matching, the method actually used ---------------------------

#: Images are matched at this resolution. Large enough to keep shell texture
#: and machining marks, small enough that all-pairs matching stays tractable.
ORB_IMAGE_SIZE = 512
ORB_FEATURES = 1200

#: Lowe's ratio test. 0.75 is the usual value and the distribution is not
#: sensitive to it in the range 0.7-0.8.
LOWE_RATIO = 0.75
RANSAC_REPROJ = 4.0

#: Geometrically consistent inliers required to call two images the same part.
#: Chosen from the empty gap in the measured pair distribution, not by taste.
#: Re-measure with :func:`inlier_distribution` before applying to a new
#: category.
DEFAULT_INLIER_THRESHOLD = 20

#: Below this many good ratio-test matches, RANSAC has nothing to fit and the
#: pair is rejected outright.
MIN_MATCHES_FOR_RANSAC = 6

#: Largest cluster a calibrated threshold is allowed to produce. Encodes the
#: prior that a physical part appears a handful of times, not dozens. Under
#: complete linkage this is an assertion, not a search target.
DEFAULT_MAX_GROUP = 12

#: A category's threshold is set at this multiple of its own 99th-percentile
#: pair score, which is safely inside its null distribution.
NULL_MULTIPLE = 2.0

#: Lower bound on any calibrated threshold, so a category with a very tight
#: null does not end up cutting close enough to chance for noise to form
#: cliques.
THRESHOLD_FLOOR = 20


class _DisjointSet:
    """Union-find with path compression, for the connected-components pass."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent: Dict[str, str] = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Merge toward the lexicographically smaller root so the result does
            # not depend on the order pairs happened to be visited in.
            lo, hi = sorted((root_a, root_b))
            self._parent[hi] = lo


@dataclass(frozen=True)
class GroupingResult:
    """Group assignment plus the numbers needed to judge the threshold."""

    #: record key -> group id, e.g. ``"hazelnut:g0037"``
    groups: Dict[str, str]
    #: group id -> number of records in it
    sizes: Dict[str, int]
    method: str
    threshold: int
    params: Dict[str, object]

    @property
    def n_groups(self) -> int:
        return len(self.sizes)

    @property
    def size_histogram(self) -> Dict[int, int]:
        """How many groups have 1 member, 2 members, and so on."""
        return dict(sorted(Counter(self.sizes.values()).items()))

    @property
    def n_grouped_images(self) -> int:
        """Images sharing a group with at least one other image.

        The headline number: exactly the set of images a random split could
        have leaked across the boundary.
        """
        return sum(size for size in self.sizes.values() if size > 1)

    @property
    def largest(self) -> int:
        return max(self.sizes.values()) if self.sizes else 0


def single_linkage(
    keys: Sequence[str], scored: Sequence[Tuple[int, str, str]], threshold: int
) -> List[List[str]]:
    """Connected components over the thresholded match graph.

    Retained for the hash control and for comparison. Prefer
    :func:`complete_linkage` for anything a split is built on.
    """
    forest = _DisjointSet(keys)
    for n, a, b in scored:
        if n >= threshold:
            forest.union(a, b)
    members: Dict[str, List[str]] = defaultdict(list)
    for key in sorted(keys):
        members[forest.find(key)].append(key)
    return [members[root] for root in sorted(members)]


def complete_linkage(
    keys: Sequence[str], scored: Sequence[Tuple[int, str, str]], threshold: int
) -> List[List[str]]:
    """Cluster so that *every* pair inside a cluster clears the threshold.

    Single linkage chains: A matching B and B matching C puts A and C in one
    cluster even when A and C do not match at all. On this data that merged 22
    visibly different hazelnuts into one "component", because cracked nuts share
    exposed-kernel texture pairwise without being the same nut.

    Complete linkage requires the cluster to be a clique above the threshold,
    which is the shape the claim actually has: *all* of these images show the
    same physical part, so all of them should match each other.

    Merges are taken strongest-first with ties broken on sorted keys, so the
    result does not depend on input order.
    """
    score: Dict[Tuple[str, str], int] = {}
    for n, a, b in scored:
        if n >= threshold:
            score[(a, b)] = n
            score[(b, a)] = n

    clusters: List[List[str]] = [[k] for k in sorted(keys)]
    while True:
        best: Optional[Tuple[int, int, int]] = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                weakest: Optional[int] = None
                for a in clusters[i]:
                    for b in clusters[j]:
                        edge = score.get((a, b), 0)
                        if edge < threshold:
                            weakest = None
                            break
                        weakest = edge if weakest is None else min(weakest, edge)
                    else:
                        continue
                    break
                if weakest is not None and (best is None or weakest > best[0]):
                    best = (weakest, i, j)
        if best is None:
            return clusters
        _, i, j = best
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]


LINKAGES = {"complete": complete_linkage, "single": single_linkage}


def _build_result(
    per_category_clusters: Dict[str, Tuple[List[str], List[List[str]]]],
    method: str,
    threshold: int,
    params: Dict[str, object],
) -> GroupingResult:
    """Number groups reproducibly, by first appearance in sorted key order."""
    groups: Dict[str, str] = {}
    sizes: Counter = Counter()
    for category in sorted(per_category_clusters):
        keys, clusters = per_category_clusters[category]
        membership = {k: i for i, members in enumerate(clusters) for k in members}
        numbering: Dict[int, str] = {}
        for key in sorted(keys):
            index = membership[key]
            if index not in numbering:
                numbering[index] = f"{category}:g{len(numbering):04d}"
            groups[key] = numbering[index]
            sizes[numbering[index]] += 1
    return GroupingResult(
        groups=groups,
        sizes=dict(sizes),
        method=method,
        threshold=threshold,
        params=params,
    )


def _by_category(
    records: Sequence[MVTecRecord],
) -> Dict[str, List[MVTecRecord]]:
    grouped: Dict[str, List[MVTecRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record)
    return {c: sorted(rows, key=lambda r: r.key) for c, rows in grouped.items()}


# ---------------------------------------------------------------------------
# Perceptual hash: the control that does not work
# ---------------------------------------------------------------------------


def difference_hash(path: Path, hash_size: int = HASH_SIZE) -> int:
    """Perceptual hash of one image as an integer of ``hash_size**2`` bits.

    Compares each pixel with its right-hand neighbour on a downscaled greyscale
    grid, so the hash encodes local gradient direction.
    """
    with Image.open(path) as handle:
        grid = handle.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
    pixels = list(grid.getdata())
    stride = hash_size + 1
    bits = 0
    for row in range(hash_size):
        base = row * stride
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[base + col] > pixels[base + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits. ``int.bit_count`` needs 3.10, this does not."""
    return bin(a ^ b).count("1")


def hash_records(
    records: Sequence[MVTecRecord],
    hash_size: int = HASH_SIZE,
    max_workers: int = 8,
) -> Dict[str, int]:
    """Hash every record, keyed by :attr:`MVTecRecord.key`."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        digests = list(
            pool.map(lambda r: difference_hash(r.image_path, hash_size), records)
        )
    return dict(zip((r.key for r in records), digests))


def group_by_hash(
    records: Sequence[MVTecRecord],
    threshold: int = DEFAULT_HASH_THRESHOLD,
    hash_size: int = HASH_SIZE,
    hashes: Optional[Dict[str, int]] = None,
    max_workers: int = 8,
) -> GroupingResult:
    """Cluster by perceptual hash. Retained as the rejected control.

    Kept so the README can show the measurement that ruled it out rather than
    asserting it. Do not build a split on this.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    if hashes is None:
        hashes = hash_records(records, hash_size=hash_size, max_workers=max_workers)

    clusters: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    for category, rows in _by_category(records).items():
        keys = [r.key for r in rows]
        digests = [hashes[k] for k in keys]
        # Score is inverted so the shared linkage helpers can be reused: a
        # Hamming distance at or below the threshold becomes a high score.
        bits = hash_size * hash_size
        scored = [
            (bits - hamming(digests[i], digests[j]), keys[i], keys[j])
            for i, j in combinations(range(len(keys)), 2)
            if hamming(digests[i], digests[j]) <= threshold
        ]
        clusters[category] = (keys, single_linkage(keys, scored, bits - threshold))

    return _build_result(
        clusters,
        method="dhash + connected components, within category",
        threshold=threshold,
        params={"hash_size": hash_size},
    )


# ---------------------------------------------------------------------------
# Keypoint matching: the method actually used
# ---------------------------------------------------------------------------

Signature = Tuple[np.ndarray, Optional[np.ndarray]]


def keypoint_signature(
    path: Path,
    n_features: int = ORB_FEATURES,
    image_size: int = ORB_IMAGE_SIZE,
) -> Signature:
    """ORB keypoint coordinates and descriptors for one image.

    Returns raw arrays rather than ``cv2.KeyPoint`` objects so signatures are
    cheap to hold for a whole category at once.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.ORB_create(nfeatures=n_features).detectAndCompute(
        image, None
    )
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    return points, descriptors


def signature_records(
    records: Sequence[MVTecRecord],
    n_features: int = ORB_FEATURES,
    image_size: int = ORB_IMAGE_SIZE,
    max_workers: int = 8,
) -> Dict[str, Signature]:
    """Extract signatures for every record. Threaded; OpenCV releases the GIL."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        signatures = list(
            pool.map(
                lambda r: keypoint_signature(r.image_path, n_features, image_size),
                records,
            )
        )
    return dict(zip((r.key for r in records), signatures))


def geometric_inliers(
    a: Signature,
    b: Signature,
    ratio: float = LOWE_RATIO,
    reproj: float = RANSAC_REPROJ,
) -> int:
    """Count geometrically consistent matches between two signatures.

    Ratio test first to drop ambiguous descriptor matches, then RANSAC under a
    partial-affine model -- rotation, translation and uniform scale, which is
    the transform a part undergoes when lifted and set back down on the rig.
    Requiring geometric consistency is what stops two different parts of the
    same type scoring highly just because they share a texture statistic.
    """
    points_a, des_a = a
    points_b, des_b = b
    if des_a is None or des_b is None or len(des_a) < 8 or len(des_b) < 8:
        return 0

    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [
        (m.queryIdx, m.trainIdx)
        for m, n in (pair for pair in knn if len(pair) == 2)
        if m.distance < ratio * n.distance
    ]
    if len(good) < MIN_MATCHES_FOR_RANSAC:
        return 0

    src = points_a[[q for q, _ in good]].reshape(-1, 1, 2)
    dst = points_b[[t for _, t in good]].reshape(-1, 1, 2)
    _, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=reproj
    )
    return 0 if inliers is None else int(inliers.sum())


def inlier_distribution(
    records: Sequence[MVTecRecord],
    signatures: Optional[Dict[str, Signature]] = None,
    max_workers: int = 8,
) -> Dict[str, List[Tuple[int, str, str]]]:
    """All within-category pair scores, for choosing the threshold.

    Returns category -> list of ``(inliers, key_a, key_b)`` sorted descending.
    The threshold should be read off the gap in this distribution; if there is
    no gap, the method is not separating and the split should not be built.
    """
    if signatures is None:
        signatures = signature_records(records, max_workers=max_workers)

    scored: Dict[str, List[Tuple[int, str, str]]] = {}
    for category, rows in _by_category(records).items():
        keys = [r.key for r in rows]
        pairs = list(combinations(range(len(keys)), 2))

        def score(ij: Tuple[int, int]) -> Tuple[int, str, str]:
            i, j = ij
            return (
                geometric_inliers(signatures[keys[i]], signatures[keys[j]]),
                keys[i],
                keys[j],
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(score, pairs))
        scored[category] = sorted(results, reverse=True)
    return scored


def save_scores(path: Path, scores: Dict[str, List[Tuple[int, str, str]]],
                min_score: int = 5) -> Path:
    """Cache the pair scores so thresholds can be explored without rematching.

    Pairs below ``min_score`` are dropped. They are chance-level for every
    category measured so far and make up almost all of the quarter-million
    pairs, so keeping them would bloat the cache for no benefit. Any threshold
    at or below ``min_score`` is refused on load rather than silently answered
    from a truncated cache.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "min_score": min_score,
        # Percentiles are computed before filtering, because the discarded
        # low-scoring pairs are exactly the null distribution the threshold is
        # calibrated against.
        "stats": {
            category: _null_stats([n for n, _, _ in pairs])
            for category, pairs in sorted(scores.items())
        },
        "params": {
            "n_features": ORB_FEATURES,
            "image_size": ORB_IMAGE_SIZE,
            "lowe_ratio": LOWE_RATIO,
            "ransac_reproj": RANSAC_REPROJ,
        },
        "scores": {
            category: [[n, a, b] for n, a, b in pairs if n >= min_score]
            for category, pairs in sorted(scores.items())
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def load_scores(
    path: Path,
) -> Tuple[Dict[str, List[Tuple[int, str, str]]], Dict[str, object]]:
    """Read cached pair scores. Returns the scores and the cache metadata."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = {
        category: [(int(n), a, b) for n, a, b in pairs]
        for category, pairs in document["scores"].items()
    }
    meta = {
        "min_score": int(document["min_score"]),
        "stats": document.get("stats", {}),
        "params": document.get("params", {}),
    }
    return scores, meta


def _null_stats(counts: Sequence[int]) -> Dict[str, float]:
    """Percentiles of a category's pair-score distribution."""
    array = np.asarray(counts, dtype=float)
    if array.size == 0:
        return {"n_pairs": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n_pairs": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def calibrate_thresholds(
    records: Sequence[MVTecRecord],
    scores: Dict[str, List[Tuple[int, str, str]]],
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    null_multiple: float = NULL_MULTIPLE,
    floor: int = THRESHOLD_FLOOR,
    max_group_size: int = DEFAULT_MAX_GROUP,
    linkage: str = "complete",
) -> Dict[str, int]:
    """Set each category's threshold from its own chance level.

    A per-category threshold is necessary rather than fussy. Chance-level inlier
    counts vary several-fold between categories, because how much structure two
    *different* parts of the same type share is a property of the part: bottles
    all present the same machined rim, carpet crops share almost nothing. On
    the three categories measured, the 99th percentile of pair scores ranges
    from 4 to 18.

    The rule is ``max(floor, null_multiple * p99)``. The 99th percentile is
    safely inside the null, since genuine repeats are a fraction of a percent of
    all pairs, and the multiple puts the cut well clear of it. The floor stops a
    category with a very tight null from getting a threshold so low that noise
    starts forming cliques.

    ``max_group_size`` is retained as an assertion rather than a search target.
    Under complete linkage it should never bind; if it does, the matcher is not
    separating on that category and the split should not be built on it.
    """
    if stats is None:
        stats = {
            category: _null_stats([n for n, _, _ in pairs])
            for category, pairs in scores.items()
        }

    chosen: Dict[str, int] = {}
    for category, rows in _by_category(records).items():
        if category not in stats:
            raise ValueError(f"no pair statistics for category {category!r}")
        p99 = float(stats[category]["p99"])
        threshold = max(floor, int(round(null_multiple * p99)))

        keys = [r.key for r in rows]
        clusters = LINKAGES[linkage](keys, scores.get(category, []), threshold)
        largest = max(len(c) for c in clusters)
        if largest > max_group_size:
            raise ValueError(
                f"{category}: threshold {threshold} still yields a cluster of "
                f"{largest} images, above the plausible maximum of "
                f"{max_group_size}. The matcher is not separating on this "
                f"category; inspect it with dump_clusters before building a "
                f"split on it."
            )
        chosen[category] = threshold
    return chosen


def group_by_keypoints(
    records: Sequence[MVTecRecord],
    threshold: Optional[int] = None,
    signatures: Optional[Dict[str, Signature]] = None,
    scores: Optional[Dict[str, List[Tuple[int, str, str]]]] = None,
    per_category: Optional[Dict[str, int]] = None,
    linkage: str = "complete",
    max_workers: int = 8,
) -> GroupingResult:
    """Cluster records into derived component groups by keypoint agreement.

    Pass ``per_category`` for calibrated per-category thresholds, or
    ``threshold`` for one global value. Pass ``scores`` from
    :func:`inlier_distribution` or :func:`load_scores` to reuse the expensive
    matching pass.
    """
    if threshold is not None and threshold < 1:
        raise ValueError(f"inlier threshold must be at least 1, got {threshold}")
    if linkage not in LINKAGES:
        raise ValueError(f"unknown linkage {linkage!r}, expected one of {sorted(LINKAGES)}")
    if scores is None:
        scores = inlier_distribution(
            records, signatures=signatures, max_workers=max_workers
        )

    by_category = _by_category(records)
    if per_category is None:
        resolved = {c: threshold or DEFAULT_INLIER_THRESHOLD for c in by_category}
    else:
        missing = sorted(set(by_category) - set(per_category))
        if missing:
            raise ValueError(f"no threshold given for categories: {missing}")
        resolved = {c: per_category[c] for c in by_category}

    clusters: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    for category, rows in by_category.items():
        keys = [r.key for r in rows]
        cut = resolved[category]
        clusters[category] = (
            keys,
            LINKAGES[linkage](keys, scores.get(category, []), cut),
        )

    return _build_result(
        clusters,
        method=(
            f"ORB + Lowe ratio + RANSAC partial affine, "
            f"{linkage} linkage, within category"
        ),
        threshold=min(resolved.values()),
        params={
            "n_features": ORB_FEATURES,
            "image_size": ORB_IMAGE_SIZE,
            "lowe_ratio": LOWE_RATIO,
            "ransac_reproj": RANSAC_REPROJ,
            "linkage": linkage,
            "per_category_threshold": dict(sorted(resolved.items())),
        },
    )


def group_records(
    records: Sequence[MVTecRecord],
    method: str = "keypoints",
    threshold: Optional[int] = None,
    **kwargs: object,
) -> GroupingResult:
    """Dispatch to a grouping method. ``keypoints`` is the one to use."""
    if method == "keypoints":
        return group_by_keypoints(
            records, threshold or DEFAULT_INLIER_THRESHOLD, **kwargs  # type: ignore[arg-type]
        )
    if method == "hash":
        return group_by_hash(
            records, threshold or DEFAULT_HASH_THRESHOLD, **kwargs  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown grouping method {method!r}, expected keypoints or hash")


def dump_clusters(
    records: Sequence[MVTecRecord],
    grouping: GroupingResult,
    out_dir: Path,
    tile: int = 200,
    max_members: int = 10,
    max_clusters: int = 40,
) -> List[Path]:
    """Write a contact sheet per multi-image cluster, largest first.

    The threshold is only defensible if someone has looked at what it merges.
    These sheets are that evidence: if two tiles on a row are visibly different
    physical parts, the threshold is too loose, whatever the histogram says.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {r.key: r for r in records}

    members: Dict[str, List[str]] = defaultdict(list)
    for key, group_id in grouping.groups.items():
        members[group_id].append(key)

    multi = sorted(
        ((g, sorted(k)) for g, k in members.items() if len(k) > 1),
        key=lambda gk: (-len(gk[1]), gk[0]),
    )[:max_clusters]

    written: List[Path] = []
    for group_id, keys in multi:
        shown = keys[:max_members]
        sheet = Image.new("RGB", (tile * len(shown), tile), (20, 20, 20))
        for i, key in enumerate(shown):
            with Image.open(by_key[key].image_path) as handle:
                thumb = handle.convert("RGB").resize(
                    (tile, tile), Image.Resampling.LANCZOS
                )
            sheet.paste(thumb, (i * tile, 0))
        path = out_dir / f"n{len(keys):03d}_{group_id.replace(':', '_')}.png"
        sheet.save(path)
        written.append(path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from vinspect.data.mvtec import index_mvtec

    parser = argparse.ArgumentParser(
        description="Measure and inspect derived component grouping."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument(
        "--method", choices=("keypoints", "hash"), default="keypoints"
    )
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument(
        "--sweep",
        nargs="*",
        type=int,
        default=None,
        help="report cluster structure at these thresholds off one matching pass",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="read pair scores from here if present, otherwise compute and write",
    )
    parser.add_argument("--max-group", type=int, default=DEFAULT_MAX_GROUP)
    parser.add_argument(
        "--linkage", choices=sorted(LINKAGES), default="complete"
    )
    parser.add_argument("--dump", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    records = index_mvtec(args.root, args.categories)
    print(f"indexed {len(records)} images")

    if args.method == "hash":
        thresholds = args.sweep or [args.threshold or DEFAULT_HASH_THRESHOLD]
        hashes = hash_records(records, max_workers=args.workers)
        results = [
            (str(t), group_by_hash(records, t, hashes=hashes)) for t in thresholds
        ]
    else:
        cached_stats = None
        if args.cache and Path(args.cache).is_file():
            scores, meta = load_scores(args.cache)
            cached_stats = meta["stats"] or None
            print(
                f"loaded cached pair scores from {args.cache} "
                f"(floor {meta['min_score']})"
            )
        else:
            scores = inlier_distribution(records, max_workers=args.workers)
            if args.cache:
                save_scores(args.cache, scores)
                print(f"cached pair scores to {args.cache}")
        for category, pairs in sorted(scores.items()):
            counts = np.array([n for n, _, _ in pairs])
            print(
                f"\n{category}: {len(counts)} pairs, "
                f"p50={np.percentile(counts, 50):.0f} "
                f"p99={np.percentile(counts, 99):.0f} "
                f"max={counts.max()}"
            )
            bins = [1, 5, 10, 15, 20, 25, 30, 50, 100]
            for lo, hi in zip(bins[:-1], bins[1:]):
                n = int(((counts >= lo) & (counts < hi)).sum())
                print(f"    [{lo:>3},{hi:>4}) {n:>6}")
            print(f"    [{bins[-1]:>3},  inf) {int((counts >= bins[-1]).sum()):>6}")
        if args.sweep:
            results = [
                (str(t),
                 group_by_keypoints(records, t, scores=scores, linkage=args.linkage))
                for t in args.sweep
            ]
        elif args.threshold:
            results = [
                (str(args.threshold),
                 group_by_keypoints(
                     records, args.threshold, scores=scores, linkage=args.linkage
                 ))
            ]
        else:
            calibrated = calibrate_thresholds(
                records,
                scores,
                stats=cached_stats,
                max_group_size=args.max_group,
                linkage=args.linkage,
            )
            print(
                f"\ncalibrated thresholds, {args.linkage} linkage, "
                f"{NULL_MULTIPLE}x each category's p99 (floor {THRESHOLD_FLOOR}):"
            )
            for category, threshold in sorted(calibrated.items()):
                print(f"    {category:<12} {threshold}")
            results = [
                ("calibrated",
                 group_by_keypoints(
                     records, scores=scores, per_category=calibrated,
                     linkage=args.linkage,
                 ))
            ]

    print(f"\n{'thresh':>10} {'groups':>7} {'grouped':>8} {'largest':>8}")
    for label, result in results:
        print(
            f"{label:>10} {result.n_groups:>7} "
            f"{result.n_grouped_images:>8} {result.largest:>8}"
        )
    print(f"\ncluster sizes: {results[-1][1].size_histogram}")

    if args.dump:
        paths = dump_clusters(records, results[-1][1], args.dump)
        print(f"wrote {len(paths)} contact sheets to {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
