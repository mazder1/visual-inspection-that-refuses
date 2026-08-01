"""The held-out-defect-class experiment.

Remove one defect class from training and calibration entirely, then show it to
the finished system and ask: does it abstain, or does it confidently guess? A
model that says *I have never seen this* is deployable; one that silently
passes an unfamiliar defect is the failure this whole project is against. The
brief calls this the single most convincing experiment in the project, and the
invisible miss in the risk-coverage run showed why: abstention tuning cannot
catch what the model cannot see, so the question is how often novel defects
land somewhere the system at least flags.

The held-out class is ``cut``, chosen because its 17 images share no derived
component with any image outside the class -- so moving them all to test
straddles nothing and disturbs no other assignment. ``crack`` and ``print``
each share components with clean training images and would have tangled the
experiment with the leakage question.

Verdicts at test time:

* **fail** -- calibrated probability >= 0.5, score inside a supported step
* **pass** -- probability < 0.5, score inside a supported step
* **no-call** -- the score falls where the calibrator has no validation
  support, or the probability sits in the middle band. Both are the system
  saying *get a human*.

For the held-out class, fail and no-call are both acceptable outcomes -- the
defect is caught either way. The failure count is the silent passes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from vinspect.data.splits import load_split, write_split
from vinspect.eval.calibration import IsotonicCalibrator

#: Middle band of calibrated probability that is nobody's verdict.
NO_CALL_BAND = (0.2, 0.8)


def build_holdout_split(
    base_split: Path, out_path: Path, category: str, defect_type: str
) -> str:
    """Variant of the grouped split with one defect class moved wholly to test.

    Refuses to build if the move would put part of a derived component in test
    while its mates stay in train or val -- that would reintroduce the leakage
    module 01 exists to prevent.
    """
    payload = load_split(base_split)
    assignments = dict(payload["assignments"])

    prefix = f"{category}/"
    held = {
        key
        for key in assignments
        if key.startswith(prefix) and key.split("/")[2] == defect_type
    }
    if not held:
        raise ValueError(f"no images of {category}/{defect_type} in the split")

    held_groups = {assignments[key]["group"] for key in held}
    mates = {
        key
        for key, value in assignments.items()
        if value["group"] in held_groups and key not in held
    }
    if mates:
        raise ValueError(
            f"moving {defect_type} would straddle components shared with "
            f"{len(mates)} other images, e.g. {sorted(mates)[:3]}"
        )

    for key in held:
        assignments[key] = {**assignments[key], "split": "test"}

    variant = {
        **payload,
        "assignments": assignments,
        "holdout": {"category": category, "defect_type": defect_type},
    }
    checksum = write_split(out_path, variant)
    return checksum


def verdicts(
    probabilities: np.ndarray,
    supported: np.ndarray,
    weak_evidence_px: Optional[np.ndarray] = None,
) -> List[str]:
    """fail / pass / no-call per image.

    The weak-evidence rule: a confident pass additionally requires **zero**
    pixels above the sensitivity level (0.33). The model emits graded evidence
    below its 0.5 decision line, and a would-be pass that still shows weak
    evidence becomes a no-call rather than a verdict. Parameter-free apart
    from the level itself, which is dev-tuned and awaits confirmation on a
    fresh held-out class.
    """
    if weak_evidence_px is None:
        weak_evidence_px = np.zeros(len(probabilities))
    out = []
    for p, s, weak in zip(probabilities, supported, weak_evidence_px):
        if not s or NO_CALL_BAND[0] <= p <= NO_CALL_BAND[1]:
            out.append("no-call")
        elif p >= 0.5:
            out.append("fail")
        elif weak > 0:
            out.append("no-call")
        else:
            out.append("pass")
    return out


def report(
    run_dir: Path, defect_type: str, passes: int = 20
) -> Dict[str, object]:
    validation = json.loads((run_dir / f"scores_val_p{passes}.json").read_text())
    test = json.loads((run_dir / f"scores_test_p{passes}.json").read_text())

    calibrator = IsotonicCalibrator().fit(
        [row["defect_score"] for row in validation],
        [row["label"] for row in validation],
    )
    scores = [row["defect_score"] for row in test]
    probabilities = calibrator.predict(scores)
    supported = calibrator.supported(scores)
    weak = np.array([row.get("weak_evidence_px", 0) for row in test])
    calls = verdicts(probabilities, supported, weak)

    def bucket(predicate) -> Dict[str, object]:
        rows = [
            (row, call)
            for row, call, keep in zip(test, calls, map(predicate, test))
            if keep
        ]
        counts = Counter(call for _, call in rows)
        return {
            "n": len(rows),
            "fail": counts["fail"],
            "no_call": counts["no-call"],
            "pass": counts["pass"],
            "silent_pass_keys": [
                row["key"] for row, call in rows if call == "pass"
            ],
        }

    held = bucket(lambda r: r["defect_type"] == defect_type)
    known = bucket(
        lambda r: r["label"] == 1 and r["defect_type"] != defect_type
    )
    clean = bucket(lambda r: r["label"] == 0)

    return {
        "held_out": held,
        "known_defects": known,
        "clean": clean,
        "n_validation": len(validation),
        "n_validation_defective": sum(r["label"] for r in validation),
        "calibrator": calibrator.summary(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out defect class experiment.")
    parser.add_argument("--category", default="hazelnut")
    parser.add_argument("--defect-type", default="cut")
    parser.add_argument("--base-split", type=Path, default=Path("splits/grouped.json"))
    parser.add_argument("--out-split", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--stage", choices=("split", "report"), required=True,
        help="split: write the variant split. report: analyse a finished run.",
    )
    args = parser.parse_args(argv)

    out_split = args.out_split or Path(
        f"splits/holdout_{args.category}_{args.defect_type}.json"
    )

    if args.stage == "split":
        checksum = build_holdout_split(
            args.base_split, out_split, args.category, args.defect_type
        )
        payload = load_split(out_split)
        moved = sum(
            1
            for key, value in payload["assignments"].items()
            if key.startswith(f"{args.category}/")
            and key.split("/")[2] == args.defect_type
            and value["split"] == "test"
        )
        print(f"wrote {out_split}")
        print(f"digest {checksum}")
        print(f"{moved} {args.defect_type} images, all in test")
        return 0

    run_dir = args.run_dir or Path(f"runs/holdout/{args.category}_grouped")
    result = report(run_dir, args.defect_type)

    print(
        f"calibrated on {result['n_validation']} validation images "
        f"({result['n_validation_defective']} defective, none of them "
        f"{args.defect_type})\n"
    )
    print(f"{'':<16} {'n':>4} {'fail':>6} {'no-call':>8} {'silent pass':>12}")
    for name, key in (
        (f"HELD OUT: {args.defect_type}", "held_out"),
        ("known defects", "known_defects"),
        ("clean", "clean"),
    ):
        row = result[key]
        print(
            f"{name:<16} {row['n']:>4} {row['fail']:>6} {row['no_call']:>8} "
            f"{row['pass']:>12}"
        )
    held = result["held_out"]
    caught = held["fail"] + held["no_call"]
    print(
        f"\nOf {held['n']} never-seen {args.defect_type} defects, {caught} are "
        f"caught ({held['fail']} failed outright, {held['no_call']} sent to a "
        f"human) and {held['pass']} pass silently."
    )
    if held["silent_pass_keys"]:
        for key in held["silent_pass_keys"]:
            print(f"    silent: {key}")

    out = run_dir / f"holdout_{args.defect_type}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
