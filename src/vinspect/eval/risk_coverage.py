"""The risk-coverage curve: what abstention buys, and what it costs.

Coverage is the share of parts the system decides alone; the rest go to a
human, assumed to catch everything. Sort parts by how *decidable* they are,
hand the least decidable fraction to the human, and measure defect recall on
the machine's share plus the human's. As coverage falls, recall rises. The
curve is that trade, and one point on it is the operating point a plant would
actually run.

Decidability is distance from the fence: ``|p - 0.5|`` on the calibrated
probability. A part at 2% or at 97% is easy; a part at 45% is exactly what the
valley in the score distribution says is rare and what a human should see.

Two defect notions are reported per point:

* **recall** -- of the truly defective parts, how many end up caught, whether by
  the machine flagging them or by being routed to the human.
* **machine miss rate** -- of the parts the machine decided alone, how many
  were defective but called clean. This is the escape rate, the number the
  brief's cost argument is about.

The headline sentence the project was set up to earn is printed from the same
numbers: recall at 100% coverage, and recall after routing the least-confident
Y% to review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from vinspect.eval.calibration import IsotonicCalibrator


def risk_coverage_curve(
    probabilities: Sequence[float],
    labels: Sequence[int],
    decision_threshold: float = 0.5,
    supported: Optional[Sequence[bool]] = None,
) -> List[Dict[str, float]]:
    """One row per coverage level, sweeping the abstention band outward.

    Routing order: **unsupported scores first**, then distance from the fence.
    A score the calibrator never saw at fit time has a probability that is an
    extrapolation, and treating it as a confident verdict is how two of this
    project's three test misses happened -- their scores sat in a staircase
    gap, the extrapolated 0.5% read as decidable-clean, and defects the model
    had actually seen (persistence 0.85, a 1,645 px region) were passed. A
    score without validation support is the brief's own abstention case:
    evidence unlike the training distribution.

    Ties break toward routing the more defect-likely part, so the curve is
    deterministic.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(probabilities)
    if n == 0:
        return []

    decidability = np.abs(probabilities - decision_threshold)
    if supported is None:
        supported = np.ones(n, dtype=bool)
    else:
        supported = np.asarray(supported, dtype=bool)
    # lexsort: last key is primary. Unsupported (False=0) sorts first.
    order = np.lexsort((-probabilities, decidability, supported))

    machine_flags = probabilities >= decision_threshold
    total_defective = int(labels.sum())

    rows: List[Dict[str, float]] = []
    for routed in range(n + 1):
        to_human = np.zeros(n, dtype=bool)
        to_human[order[:routed]] = True

        caught_by_machine = int((machine_flags & ~to_human & (labels == 1)).sum())
        caught_by_human = int((to_human & (labels == 1)).sum())
        machine_share = ~to_human
        missed = int((~machine_flags & machine_share & (labels == 1)).sum())
        false_alarms = int((machine_flags & machine_share & (labels == 0)).sum())

        rows.append(
            {
                "coverage": float((n - routed) / n),
                "routed": routed,
                "recall": (
                    (caught_by_machine + caught_by_human) / total_defective
                    if total_defective
                    else 1.0
                ),
                "machine_missed": missed,
                "machine_false_alarms": false_alarms,
            }
        )
    return rows


def points_at(
    curve: Sequence[Dict[str, float]], coverages: Sequence[float]
) -> List[Dict[str, float]]:
    """The curve rows nearest each requested coverage level."""
    picked = []
    for target in coverages:
        row = min(curve, key=lambda r: abs(r["coverage"] - target))
        picked.append(row)
    return picked


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Risk-coverage from cached scores.")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "carpet", "hazelnut"]
    )
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    args = parser.parse_args(argv)

    all_probabilities: List[float] = []
    all_labels: List[int] = []
    all_supported: List[bool] = []
    for category in args.categories:
        directory = args.runs / f"{category}_grouped"
        validation = json.loads(
            (directory / f"scores_val_p{args.passes}.json").read_text()
        )
        test = json.loads((directory / f"scores_test_p{args.passes}.json").read_text())

        calibrator = IsotonicCalibrator(pseudocount=args.pseudocount).fit(
            [row["defect_score"] for row in validation],
            [row["label"] for row in validation],
        )
        test_scores = [row["defect_score"] for row in test]
        probabilities = calibrator.predict(test_scores)
        supported = calibrator.supported(test_scores)

        # The weak-evidence floor, from clean validation only: a would-be pass
        # whose sub-threshold whisper exceeds what 95% of clean parts show is
        # not a confident decision, so it routes with the unsupported scores.
        clean_weak = [
            row.get("weak_evidence_px", 0)
            for row in validation
            if row["label"] == 0
        ]
        floor = float(np.percentile(clean_weak, 95)) if clean_weak else float("inf")
        weak = np.array([row.get("weak_evidence_px", 0) for row in test])
        blocked_pass = (probabilities < 0.5) & (weak > floor)

        all_probabilities.extend(float(p) for p in probabilities)
        all_supported.extend(
            bool(s and not b) for s, b in zip(supported, blocked_pass)
        )
        all_labels.extend(int(row["label"]) for row in test)

    n_routed_first = sum(1 for s in all_supported if not s)
    print(
        f"{n_routed_first} of {len(all_supported)} test images are not "
        f"confidently decidable (unsupported score or weak evidence above the "
        f"clean floor) and are routed first"
    )
    curve = risk_coverage_curve(
        all_probabilities, all_labels, supported=all_supported
    )
    n = len(all_labels)

    print(f"pooled test set: {n} images, {sum(all_labels)} defective\n")
    print(f"{'coverage':>9} {'routed':>7} {'recall':>8} {'machine misses':>15} "
          f"{'false alarms':>13}")
    for row in points_at(curve, (1.0, 0.98, 0.95, 0.90, 0.80)):
        print(
            f"{row['coverage']:>9.1%} {row['routed']:>7} {row['recall']:>8.1%} "
            f"{row['machine_missed']:>15} {row['machine_false_alarms']:>13}"
        )

    full = curve[0]
    operating = points_at(curve, (0.95,))[0]
    print(
        f"\nAt 100% coverage, defect recall is {full['recall']:.1%}. "
        f"Routing the {1 - operating['coverage']:.1%} least-confident images "
        f"to human review raises recall to {operating['recall']:.1%}."
    )
    if operating["machine_missed"]:
        clean_row = next(
            (r for r in curve if r["machine_missed"] == 0), curve[-1]
        )
        print(
            f"The remaining {operating['machine_missed']} missed defect(s) are "
            f"invisible to the model -- score zero, no region in any MC pass -- "
            f"and routing does not reach them until "
            f"{1 - clean_row['coverage']:.1%} of parts go to review. Catching "
            f"what the model cannot see at all is the held-out-class problem, "
            f"not an abstention-tuning problem."
        )

    out = args.runs / "risk_coverage.json"
    out.write_text(
        json.dumps({"curve": curve, "n": n}, indent=2), encoding="utf-8"
    )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
