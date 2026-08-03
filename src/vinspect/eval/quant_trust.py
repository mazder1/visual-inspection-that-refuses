"""Stage 3 of the quantisation experiment: what compression did to trust.

The comparison chain has three rungs, and the middle one is what makes the
attribution clean::

    torch fp32 (published)  ->  ONNX fp32  ->  ONNX INT8

Comparing INT8 straight against the published numbers would blend two causes,
the engine swap and the compression. So everything is first rescored with the
fp32 ONNX graph under the identical protocol: its deviation from the published
numbers is the measured noise floor (20-pass MC statistics wobble; the
within-engine control in stage 1 put the spread at up to 3.2x on std maps).
An INT8 delta must clear that floor, not zero, to be called drift.

Two arms for the INT8 rung:

* **Arm A, frozen:** the calibrator and floor fitted on the torch fp32
  validation scores, applied unchanged to INT8 scores -- what a team does when
  it compresses a shipped model and assumes the calibration still holds.
  Measures drift.
* **Arm B, refit:** staircase and floor refitted on the INT8 model's own
  validation scores -- validation doing exactly the job it is reserved for.
  Measures whether recalibration repairs the drift. Repair working means the
  damage was a monotone shift; repair failing means scores were reordered,
  which no output-side fix can undo.

The holdout verdict tables are a hard gate independent of Brier: if the
never-seen-class catches degrade, INT8 does not ship at any speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from vinspect.data.mvtec import MVTecDataset, index_mvtec
from vinspect.data.splits import load_split, records_for_split
from vinspect.eval.calibration import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
)
from vinspect.eval.holdout import verdicts
from vinspect.uncertainty.mc_dropout import MCPrediction
from vinspect.uncertainty.regions import score_image

PASSES = 20
WEAK_THRESHOLD = 0.33
SCORES_DIR = Path("runs/quant/scores")

#: Every model/dataset pair in the study. name -> (run_dir, category, split).
WORK: Dict[str, Dict[str, str]] = {
    "bottle": {"run": "runs", "category": "bottle", "split": "splits/grouped.json", "bundle": "bundles/bottle"},
    "carpet": {"run": "runs", "category": "carpet", "split": "splits/grouped.json", "bundle": "bundles/carpet"},
    "hazelnut": {"run": "runs", "category": "hazelnut", "split": "splits/grouped.json", "bundle": "bundles/hazelnut"},
    "holdout_cut": {"run": "runs/holdout", "category": "hazelnut", "split": "splits/holdout_hazelnut_cut.json", "bundle": "bundles/holdout_cut/hazelnut", "held": "cut"},
    "holdout_hole": {"run": "runs/holdout_carpet", "category": "carpet", "split": "splits/holdout_carpet_hole.json", "bundle": "bundles/holdout_hole/carpet", "held": "hole"},
}
ENGINE_FILES = {"fp32": "model.onnx", "int8": "model.int8.onnx"}


def score_records_with_onnx(
    onnx_path: Path,
    records: Sequence,
    image_size: int,
    threshold: float,
    passes: int = PASSES,
) -> List[Dict[str, object]]:
    """The serving chain's scoring, with an ONNX session as the engine.

    Produces rows in exactly the format of ``abstain.score_split`` so every
    existing analysis runs unchanged on top.
    """
    from vinspect.serve.onnx_export import make_session

    session = make_session(onnx_path)
    dataset = MVTecDataset(records, image_size=image_size)

    rows: List[Dict[str, object]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        array = sample["image"].numpy()[None].astype(np.float32)
        stack = np.stack(
            [
                1.0 / (1.0 + np.exp(-session.run(None, {"image": array})[0][0, 0]))
                for _ in range(passes)
            ]
        )
        tensor = torch.from_numpy(stack.astype(np.float32))
        prediction = MCPrediction(
            mean=tensor.mean(dim=0), std=tensor.std(dim=0), passes=tensor
        )
        image_score = score_image(
            prediction, threshold=threshold, weak_threshold=WEAK_THRESHOLD
        )
        rows.append(
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
                "weak_evidence_px": int(
                    (prediction.mean.numpy() > WEAK_THRESHOLD).sum()
                ),
            }
        )
    return rows


def run_scoring(engine: str, names: Optional[List[str]] = None) -> None:
    """Score every work item's val and test with one engine, caching to disk."""
    for name, work in WORK.items():
        if names and name not in names:
            continue
        chain = json.loads(
            (Path(work["bundle"]) / "chain.json").read_text(encoding="utf-8")
        )
        image_size = chain["model"]["image_size"]
        threshold = chain["chain"]["threshold"]
        onnx_path = Path(work["bundle"]) / ENGINE_FILES[engine]

        payload = load_split(Path(work["split"]))
        records = index_mvtec(Path("data/mvtec_ad"), [work["category"]])
        for split in ("val", "test"):
            out = SCORES_DIR / engine / f"{name}_{split}.json"
            if out.is_file():
                print(f"cached: {out}")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            rows = score_records_with_onnx(
                onnx_path,
                records_for_split(records, payload, split),
                image_size,
                threshold,
            )
            out.write_text(json.dumps(rows), encoding="utf-8")
            print(f"scored {name} {split} ({len(rows)} images) -> {out}")


# --- analysis --------------------------------------------------------------


def _torch_rows(name: str, split: str) -> List[Dict[str, object]]:
    work = WORK[name]
    path = Path(work["run"]) / f"{work['category']}_grouped" / f"scores_{split}_p{PASSES}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _engine_rows(engine: str, name: str, split: str) -> List[Dict[str, object]]:
    if engine == "torch":
        return _torch_rows(name, split)
    return json.loads(
        (SCORES_DIR / engine / f"{name}_{split}.json").read_text(encoding="utf-8")
    )


def _chain_from(validation: List[Dict[str, object]]):
    calibrator = IsotonicCalibrator().fit(
        [row["defect_score"] for row in validation],
        [row["label"] for row in validation],
    )
    clean_weak = [
        row["weak_evidence_px"] for row in validation if row["label"] == 0
    ]
    floor = float(np.percentile(clean_weak, 95)) if clean_weak else float("inf")
    return calibrator, floor


def _apply(calibrator, floor, test: List[Dict[str, object]]):
    scores = [row["defect_score"] for row in test]
    probabilities = calibrator.predict(scores)
    supported = calibrator.supported(scores)
    weak = np.array([row["weak_evidence_px"] for row in test])
    calls = verdicts(probabilities, supported, weak, floor)
    return probabilities, calls


def arm_metrics(
    engine: str, calibration_engine: str, names: Sequence[str]
) -> Dict[str, object]:
    """One arm: scores from ``engine``, chain fitted on ``calibration_engine``."""
    pooled_p: List[float] = []
    pooled_labels: List[int] = []
    counts = {"fail": 0, "pass": 0, "no-call": 0}
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for name in names:
        calibrator, floor = _chain_from(_engine_rows(calibration_engine, name, "val"))
        test = _engine_rows(engine, name, "test")
        probabilities, calls = _apply(calibrator, floor, test)
        labels = [int(row["label"]) for row in test]

        pooled_p.extend(float(p) for p in probabilities)
        pooled_labels.extend(labels)
        for call, label in zip(calls, labels):
            counts[call] += 1
            if call == "no-call":
                # Human resolves it correctly: counted into the confusion as
                # caught, consistent with the published system-incl-human view.
                confusion["tp" if label else "tn"] += 1
            elif call == "fail":
                confusion["tp" if label else "fp"] += 1
            else:
                confusion["fn" if label else "tn"] += 1

    labels_array = np.array(pooled_labels)
    p_array = np.array(pooled_p)
    n = len(labels_array)
    return {
        "n": n,
        "brier": brier_score(p_array, labels_array),
        "ece": expected_calibration_error(p_array, labels_array),
        "verdicts": counts,
        "review_rate": counts["no-call"] / n if n else 0.0,
        "confusion_incl_human": confusion,
        "silent_misses": confusion["fn"],
    }


def holdout_table(engine: str, calibration_engine: str, name: str) -> Dict[str, int]:
    work = WORK[name]
    calibrator, floor = _chain_from(_engine_rows(calibration_engine, name, "val"))
    test = _engine_rows(engine, name, "test")
    _, calls = _apply(calibrator, floor, test)

    held = work["held"]
    table = {"fail": 0, "no-call": 0, "silent_pass": 0}
    for row, call in zip(test, calls):
        if row["defect_type"] != held:
            continue
        if call == "fail":
            table["fail"] += 1
        elif call == "no-call":
            table["no-call"] += 1
        else:
            table["silent_pass"] += 1
    return table


MAINS = ("bottle", "carpet", "hazelnut")


def report() -> None:
    arms = [
        ("torch published", "torch", "torch"),
        ("fp32 onnx, frozen chain", "fp32", "torch"),
        ("int8, frozen chain (drift)", "int8", "torch"),
        ("int8, refit chain (repair)", "int8", "int8"),
    ]
    print("=== pooled main test set (237 images, 44 defective) ===")
    print(f"{'arm':<30} {'brier':>8} {'ece':>7} {'no-call':>8} {'fail':>5} "
          f"{'pass':>5} {'silent FN':>10}")
    for label, engine, calibration in arms:
        m = arm_metrics(engine, calibration, MAINS)
        print(
            f"{label:<30} {m['brier']:>8.4f} {m['ece']:>7.4f} "
            f"{m['verdicts']['no-call']:>8} {m['verdicts']['fail']:>5} "
            f"{m['verdicts']['pass']:>5} {m['silent_misses']:>10}"
        )

    for name, total in (("holdout_hole", 17), ("holdout_cut", 17)):
        print(f"\n=== {name} ({WORK[name]['held']}, {total} never-seen) ===")
        print(f"{'arm':<30} {'fail':>5} {'no-call':>8} {'SILENT':>7}")
        for label, engine, calibration in arms:
            table = holdout_table(engine, calibration, name)
            print(
                f"{label:<30} {table['fail']:>5} {table['no-call']:>8} "
                f"{table['silent_pass']:>7}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quantisation trust study.")
    parser.add_argument("--stage", choices=("score", "report"), required=True)
    parser.add_argument("--engine", choices=("fp32", "int8"), default=None)
    parser.add_argument("--names", nargs="*", default=None)
    args = parser.parse_args(argv)

    if args.stage == "score":
        engines = [args.engine] if args.engine else ["fp32", "int8"]
        for engine in engines:
            run_scoring(engine, args.names)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
