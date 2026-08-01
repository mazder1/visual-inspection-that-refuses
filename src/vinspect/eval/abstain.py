"""The module 03 pipeline: MC-dropout scores -> calibration -> honest report.

Per category: score every validation and test image with MC dropout and
region-level scoring, fit the isotonic calibrator on validation outcomes, and
report Brier, ECE and the reliability table on test. Scores are cached to JSON
so the expensive part runs once.

Calibration is fitted **per category**, because the raw scores are not
comparable across categories -- a bottle defect region is an order of magnitude
larger than a carpet thread, so one shared staircase would mostly be learning
which category an image came from. The calibrated probabilities, unlike the
scores, share a scale and can be pooled for the overall report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from vinspect.data.mvtec import MVTecDataset, index_mvtec
from vinspect.data.splits import load_split, records_for_split
from vinspect.eval.calibration import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from vinspect.models.unet import UNet
from vinspect.uncertainty.mc_dropout import mc_predict_batch
from vinspect.uncertainty.regions import score_image

#: Sensitivity level for the weak-evidence check. Below the 0.5 decision line,
#: the model still emits graded evidence; at 0.33 the inspected clean panels
#: showed at most 7 px while faint novel defects showed hundreds. Chosen while
#: looking at the held-out cuts, so treat any number derived from them as
#: dev-set performance until confirmed on a fresh held-out class.
WEAK_EVIDENCE_THRESHOLD = 0.33


def score_split(
    checkpoint_path: Path,
    split: str,
    passes: int = 20,
    batch_size: int = 4,
    device: str = "cuda",
) -> List[Dict[str, object]]:
    """MC-dropout region scores for every image of one split, one category."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = device if torch.cuda.is_available() else "cpu"

    model = UNet(
        base_channels=config["base_channels"],
        depth=config["depth"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    payload = load_split(Path(config["split_path"]))
    records = index_mvtec(Path(config["root"]), [config["category"]])
    rows = records_for_split(records, payload, split)
    dataset = MVTecDataset(rows, image_size=config["image_size"])

    results = []
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([sample["image"] for sample in batch])
        predictions = mc_predict_batch(model, images, passes=passes)
        for sample, prediction in zip(batch, predictions):
            # Hysteresis extraction: weak pixels bridge strong fragments into
            # one region but contribute no evidence themselves. Verified on dev
            # to merge fragmented defect evidence at zero cost to clean parts.
            image_score = score_image(
                prediction,
                threshold=config["threshold"],
                weak_threshold=WEAK_EVIDENCE_THRESHOLD,
            )
            weak_pixels = int(
                (prediction.mean > WEAK_EVIDENCE_THRESHOLD).sum()
            )
            results.append(
                {
                    "key": sample["key"],
                    "label": int(sample["label"]),
                    "defect_type": sample["defect_type"],
                    "defect_score": image_score.defect_score,
                    "mass_score": image_score.mass_score,
                    "n_regions": image_score.n_regions,
                    "largest_area": image_score.largest_area,
                    "largest_extent": image_score.largest_extent,
                    "persistence": image_score.persistence,
                    "area_cv": image_score.area_cv,
                    "interior_std": image_score.interior_std,
                    "weak_evidence_px": weak_pixels,
                }
            )
    return results


def _cached_scores(
    run_dir: Path, category: str, split: str, passes: int, refresh: bool
) -> List[Dict[str, object]]:
    cache = run_dir / f"{category}_grouped" / f"scores_{split}_p{passes}.json"
    if cache.is_file() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    results = score_split(run_dir / f"{category}_grouped" / "best.pt", split, passes)
    cache.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate region scores and report on test."
    )
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "carpet", "hazelnut"]
    )
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    pooled_probabilities: List[float] = []
    pooled_labels: List[int] = []
    report: Dict[str, object] = {}

    for category in args.categories:
        validation = _cached_scores(args.runs, category, "val", args.passes, args.refresh)
        test = _cached_scores(args.runs, category, "test", args.passes, args.refresh)

        calibrator = IsotonicCalibrator(pseudocount=args.pseudocount).fit(
            [row["defect_score"] for row in validation],
            [row["label"] for row in validation],
        )
        probabilities = calibrator.predict([row["defect_score"] for row in test])
        labels = np.array([row["label"] for row in test])

        # The do-nothing baseline: always claim the validation base rate.
        baseline = brier_score(
            np.full(len(labels), calibrator.base_rate), labels
        )
        brier = brier_score(probabilities, labels)
        ece = expected_calibration_error(probabilities, labels)

        pooled_probabilities.extend(float(p) for p in probabilities)
        pooled_labels.extend(int(v) for v in labels)

        print(f"\n=== {category} ===")
        print(
            f"calibrator: {len(calibrator.steps)} steps from "
            f"{len(validation)} validation images "
            f"(base rate {calibrator.base_rate:.1%})"
        )
        print(
            f"test Brier {brier:.4f} against always-base-rate {baseline:.4f}; "
            f"ECE {ece:.4f}"
        )
        for step in calibrator.steps:
            print(
                f"    score {step.lower:>8.1f} .. {step.upper:>8.1f} "
                f"-> {step.probability:5.1%}  ({step.count} images, "
                f"raw {step.raw_rate:.1%})"
            )
        report[category] = {
            "brier": brier,
            "brier_base_rate": baseline,
            "ece": ece,
            "calibrator": calibrator.summary(),
        }

    pooled = np.asarray(pooled_probabilities)
    labels = np.asarray(pooled_labels)
    print(f"\n=== pooled test reliability ({len(pooled)} images) ===")
    print(f"{'claimed':>8} {'observed':>9} {'n':>5}")
    rows = reliability_bins(pooled, labels, n_bins=8)
    for row in rows:
        print(f"{row['claimed']:>8.1%} {row['observed']:>9.1%} {row['count']:>5}")
    print(
        f"pooled Brier {brier_score(pooled, labels):.4f}, "
        f"ECE {expected_calibration_error(pooled, labels):.4f}"
    )
    report["pooled"] = {
        "brier": brier_score(pooled, labels),
        "ece": expected_calibration_error(pooled, labels),
        "reliability": rows,
    }

    out = args.runs / "calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
