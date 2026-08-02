"""The split generator. Written before the model, on purpose.

Produces a three-way train/val/test split in which no derived component group
straddles a boundary, and a random split at the same ratios for contrast. The
gap between the two is a reported result: it measures how much a naive
evaluation would have overstated the model.

Three things this module is careful about.

**It ignores MVTec's shipped train/test directories.** That division puts every
defective image in ``test`` and none in ``train``, so it cannot support a
three-way split with defectives in val and test. The whole pool is re-split.

**It packs on defectives first.** Clean images are plentiful; defective ones are
not, and recall, calibration and the risk-coverage curve are all measured on
them. Matching the target defective count per split therefore takes priority
over matching the target total.

**The random baseline is stratified the same way.** It differs from the grouped
split in exactly one respect, that it ignores group membership, so the delta
between the two is attributable to leakage rather than to some other difference
in how they were built.

The artifact is content-hashed, so a result can be tied to the split it was
measured on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from vinspect.data.grouping import (
    DEFAULT_MAX_GROUP,
    GroupingResult,
    calibrate_thresholds,
    group_by_keypoints,
    inlier_distribution,
    load_scores,
    save_scores,
)
from vinspect.data.mvtec import MVTecRecord, index_mvtec

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
DEFAULT_RATIOS: Dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
SCHEMA_VERSION = 1


class SplitError(RuntimeError):
    """The split is invalid, or an artifact does not match its digest."""


def _validate_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    if set(ratios) != set(SPLITS):
        raise SplitError(f"ratios must cover exactly {SPLITS}, got {sorted(ratios)}")
    if any(v <= 0 for v in ratios.values()):
        raise SplitError(f"every ratio must be positive, got {ratios}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise SplitError(f"ratios must sum to 1.0, got {total}")
    return dict(ratios)


def assign_grouped(
    records: Sequence[MVTecRecord],
    grouping: GroupingResult,
    ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Assign whole groups to splits, category by category.

    Greedy largest-first packing against two targets, defectives before totals.
    Groups are placed in descending order of defect count so the scarce, hard to
    balance items are committed while every split still has room; the tail of
    clean-only groups then fills whatever deficit is left.

    Deterministic: no RNG is involved. Ordering is fixed by defect count, size
    and group id, so the same records and grouping always give the same split.
    """
    ratios = _validate_ratios(ratios or DEFAULT_RATIOS)

    by_category: Dict[str, List[MVTecRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)

    assignments: Dict[str, str] = {}
    for category in sorted(by_category):
        rows = by_category[category]
        members: Dict[str, List[MVTecRecord]] = defaultdict(list)
        for record in rows:
            members[grouping.groups[record.key]].append(record)

        n_total = len(rows)
        n_defective = sum(r.label for r in rows)
        target_total = {s: ratios[s] * n_total for s in SPLITS}
        target_defective = {s: ratios[s] * n_defective for s in SPLITS}
        have_total = {s: 0 for s in SPLITS}
        have_defective = {s: 0 for s in SPLITS}

        ordered = sorted(
            members.items(),
            key=lambda kv: (-sum(r.label for r in kv[1]), -len(kv[1]), kv[0]),
        )
        for group_id, group_records_ in ordered:
            group_defective = sum(r.label for r in group_records_)
            if group_defective:
                # Defectives are the scarce resource, so place them against the
                # defective target first and use the total only to break ties.
                chosen = max(
                    SPLITS,
                    key=lambda s: (
                        target_defective[s] - have_defective[s],
                        target_total[s] - have_total[s],
                        -SPLITS.index(s),
                    ),
                )
            else:
                # A clean-only group cannot reduce anyone's defective deficit.
                # Ranking it on that deficit would pin every remaining clean
                # group to whichever split is most defective-starved, which is
                # how this quietly produced a 47/18/35 split before.
                chosen = max(
                    SPLITS,
                    key=lambda s: (
                        target_total[s] - have_total[s],
                        -SPLITS.index(s),
                    ),
                )
            have_total[chosen] += len(group_records_)
            have_defective[chosen] += group_defective
            for record in group_records_:
                assignments[record.key] = chosen

    return assignments


def assign_random(
    records: Sequence[MVTecRecord],
    ratios: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> Dict[str, str]:
    """The naive baseline: same ratios and stratification, no group constraint.

    Stratified by category and label so it is not handicapped by an unlucky
    draw. This is deliberately the *strongest* random split, not a strawman.
    Whatever gap remains against the grouped split is attributable to leakage.
    """
    ratios = _validate_ratios(ratios or DEFAULT_RATIOS)
    rng = random.Random(seed)

    strata: Dict[Tuple[str, int], List[MVTecRecord]] = defaultdict(list)
    for record in records:
        strata[(record.category, record.label)].append(record)

    assignments: Dict[str, str] = {}
    for stratum in sorted(strata):
        rows = sorted(strata[stratum], key=lambda r: r.key)
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(ratios["train"] * n))
        n_val = int(round(ratios["val"] * n))
        # Give the remainder to test rather than letting rounding drop records.
        bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val)}
        for split, (lo, hi) in bounds.items():
            for record in rows[lo:hi]:
                assignments[record.key] = split
        for record in rows[n_train + n_val :]:
            assignments[record.key] = "test"

    return assignments


def verify_no_leakage(
    assignments: Dict[str, str], grouping: GroupingResult
) -> Dict[str, object]:
    """Confirm no group straddles a split boundary.

    Called on every generated split and asserted in the tests. A split that has
    not been checked for leakage is a split that leaks.
    """
    seen: Dict[str, set] = defaultdict(set)
    for key, split in assignments.items():
        seen[grouping.groups[key]].add(split)
    straddling = sorted(g for g, splits in seen.items() if len(splits) > 1)
    return {
        "clean": not straddling,
        "n_groups": len(seen),
        "n_straddling": len(straddling),
        "straddling": straddling[:20],
    }


def build_artifact(
    records: Sequence[MVTecRecord],
    assignments: Dict[str, str],
    grouping: GroupingResult,
    kind: str,
    ratios: Dict[str, float],
    seed: int,
    dataset_root: Path,
) -> Dict[str, object]:
    """Assemble the split payload. Absolute paths are excluded on purpose.

    Record keys are relative to the dataset root, so the digest identifies the
    split's *membership* and does not change when the data is mounted somewhere
    else.
    """
    if kind not in ("grouped", "random"):
        raise SplitError(f"unknown split kind: {kind!r}")
    missing = sorted({r.key for r in records} - set(assignments))
    if missing:
        raise SplitError(
            f"{len(missing)} records were never assigned, e.g. {missing[:5]}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "ratios": dict(sorted(ratios.items())),
        "seed": seed,
        "dataset": {
            "name": "MVTec AD",
            "root_basename": Path(dataset_root).name,
            "licence": "CC BY-NC-SA 4.0, non-commercial research use",
        },
        "categories": sorted({r.category for r in records}),
        "n_images": len(records),
        "grouping": {
            "method": grouping.method,
            "threshold": grouping.threshold,
            "params": dict(sorted(grouping.params.items())),
            "n_groups": grouping.n_groups,
            "n_images_in_multi_image_groups": grouping.n_grouped_images,
            "largest_group": grouping.largest,
            "size_histogram": {str(k): v for k, v in grouping.size_histogram.items()},
        },
        "assignments": {
            key: {"split": assignments[key], "group": grouping.groups[key]}
            for key in sorted(assignments)
        },
    }


def digest(payload: Dict[str, object]) -> str:
    """SHA-256 over the canonical JSON encoding of the payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_split(path: Path, payload: Dict[str, object]) -> str:
    """Write the artifact with its digest alongside. Returns the digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = digest(payload)
    document = {"digest": checksum, "payload": payload}
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    path.with_suffix(".sha256").write_text(
        f"{checksum}  {path.name}\n", encoding="utf-8"
    )
    return checksum


def load_split(path: Path) -> Dict[str, object]:
    """Read a split artifact, refusing it if the digest does not match.

    An edited split file is worse than a missing one, because results already
    reported against it silently stop meaning what they said.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    recomputed = digest(payload)
    if recomputed != document["digest"]:
        raise SplitError(
            f"digest mismatch for {path}: recorded {document['digest']}, "
            f"recomputed {recomputed}. The split file has been modified."
        )
    return payload


def records_for_split(
    records: Sequence[MVTecRecord], payload: Dict[str, object], split: str
) -> List[MVTecRecord]:
    """Select the records belonging to one split of a loaded artifact."""
    if split not in SPLITS:
        raise SplitError(f"unknown split {split!r}, expected one of {SPLITS}")
    assignments: Dict[str, Dict[str, str]] = payload["assignments"]  # type: ignore[assignment]
    unknown = sorted({r.key for r in records} - set(assignments))
    if unknown:
        raise SplitError(
            f"{len(unknown)} indexed records are absent from the split artifact, "
            f"e.g. {unknown[:5]}. The index and the split disagree about the data."
        )
    return [r for r in records if assignments[r.key]["split"] == split]


def format_split_report(
    records: Sequence[MVTecRecord], assignments: Dict[str, str]
) -> str:
    """Per-category, per-split counts. Defectives are the column that matters."""
    by_key = {r.key: r for r in records}
    header = (
        f"{'category':<12} {'split':<6} {'total':>6} {'clean':>6} "
        f"{'defect':>7} {'types':>6}"
    )
    lines = [header, "-" * len(header)]
    for category in sorted({r.category for r in records}):
        for split in SPLITS:
            rows = [
                by_key[k]
                for k, s in assignments.items()
                if s == split and by_key[k].category == category
            ]
            defective = [r for r in rows if r.label == 1]
            lines.append(
                f"{category:<12} {split:<6} {len(rows):>6} "
                f"{len(rows) - len(defective):>6} {len(defective):>7} "
                f"{len({r.defect_type for r in defective}):>6}"
            )
    lines.append("-" * len(header))
    for split in SPLITS:
        rows = [by_key[k] for k, s in assignments.items() if s == split]
        defective = sum(r.label for r in rows)
        share = len(rows) / max(len(records), 1)
        lines.append(
            f"{'ALL':<12} {split:<6} {len(rows):>6} "
            f"{len(rows) - defective:>6} {defective:>7} "
            f"{'':>6}  ({share:.1%} of images)"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the grouped and random splits, and hash them."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("splits"))
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="one global inlier threshold; omit to calibrate per category",
    )
    parser.add_argument("--max-group", type=int, default=DEFAULT_MAX_GROUP)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="read pair scores from here if present, otherwise compute and write",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[DEFAULT_RATIOS[s] for s in SPLITS],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    args = parser.parse_args(argv)

    ratios = _validate_ratios(dict(zip(SPLITS, args.ratios)))
    records = index_mvtec(args.root, args.categories)
    print(
        f"indexed {len(records)} images across "
        f"{len({r.category for r in records})} categories"
    )

    cached_stats = None
    if args.cache and Path(args.cache).is_file():
        scores, meta = load_scores(args.cache)
        cached_stats = meta["stats"] or None
        print(f"loaded cached pair scores from {args.cache}")
    else:
        scores = inlier_distribution(records, max_workers=args.workers)
        if args.cache:
            save_scores(args.cache, scores)
            print(f"cached pair scores to {args.cache}")

    if args.threshold:
        grouping = group_by_keypoints(records, args.threshold, scores=scores)
    else:
        per_category = calibrate_thresholds(
            records, scores, stats=cached_stats, max_group_size=args.max_group
        )
        print(f"calibrated thresholds: {per_category}")
        grouping = group_by_keypoints(
            records, scores=scores, per_category=per_category
        )

    print(
        f"grouped into {grouping.n_groups} components; "
        f"{grouping.n_grouped_images} images sit in a group with at least one "
        f"other image, largest group {grouping.largest}"
    )
    print(f"cluster sizes (size: count): {grouping.size_histogram}")

    # The random split is measured against the same grouping, purely so its
    # leakage can be reported against one consistent definition of component.
    for kind, assignments in (
        ("grouped", assign_grouped(records, grouping, ratios)),
        ("random", assign_random(records, ratios, seed=args.seed)),
    ):
        leakage = verify_no_leakage(assignments, grouping)
        if kind == "grouped" and not leakage["clean"]:
            raise SplitError(
                f"grouped split leaks: {leakage['n_straddling']} groups straddle "
                f"a boundary, e.g. {leakage['straddling'][:5]}"
            )
        payload = build_artifact(
            records, assignments, grouping, kind, ratios, args.seed, args.root
        )
        checksum = write_split(Path(args.out) / f"{kind}.json", payload)

        print(f"\n=== {kind} split ===")
        print(format_split_report(records, assignments))
        print(
            f"components straddling a boundary: {leakage['n_straddling']} "
            f"of {leakage['n_groups']}"
        )
        print(f"digest: {checksum}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
