"""Bundle everything the service needs into a self-contained directory.

The service must not depend on the dataset, the runs directory, or GPU-side
code. Per category the bundle holds the model weights and one ``chain.json``
carrying the calibrator's fitted steps, the weak-evidence floor, the
thresholds, and the digests that tie the bundle back to the split and
checkpoint it came from. Serving never refits anything: the chain is frozen at
export time, exactly as evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from vinspect.eval.abstain import WEAK_EVIDENCE_THRESHOLD
from vinspect.eval.calibration import IsotonicCalibrator
from vinspect.eval.holdout import NO_CALL_BAND


def export_category(
    run_dir: Path, category: str, out_dir: Path, passes: int = 20
) -> Path:
    source = run_dir / f"{category}_grouped"
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    validation = json.loads((source / f"scores_val_p{passes}.json").read_text())

    calibrator = IsotonicCalibrator().fit(
        [row["defect_score"] for row in validation],
        [row["label"] for row in validation],
    )
    clean_weak = [
        row["weak_evidence_px"] for row in validation if row["label"] == 0
    ]
    floor = float(np.percentile(clean_weak, 95))

    destination = out_dir / category
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best.pt", destination / "model.pt")
    weights_digest = hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest()

    chain = {
        "category": category,
        "model": {
            "base_channels": checkpoint["config"]["base_channels"],
            "depth": checkpoint["config"]["depth"],
            "dropout": checkpoint["config"]["dropout"],
            "image_size": checkpoint["config"]["image_size"],
            "weights_sha256": weights_digest,
        },
        "chain": {
            "mc_passes": passes,
            "threshold": checkpoint["config"]["threshold"],
            "weak_threshold": WEAK_EVIDENCE_THRESHOLD,
            "weak_floor": floor,
            "no_call_band": list(NO_CALL_BAND),
            "calibrator": calibrator.summary(),
        },
        "provenance": {
            "split_digest": checkpoint["split_digest"],
            "split_kind": checkpoint["split_kind"],
            "trained_epoch": checkpoint["epoch"],
            "val_dice": checkpoint["val_dice"],
            "n_validation": len(validation),
            "n_validation_defective": int(
                sum(row["label"] for row in validation)
            ),
        },
        "disclaimer": (
            "Decision support only. Routes parts to an inspector; never makes "
            "a final accept or reject decision on its own. Not qualified for "
            "production use."
        ),
    }
    (destination / "chain.json").write_text(
        json.dumps(chain, indent=2), encoding="utf-8"
    )
    return destination


def refit_chain_for_int8(
    bundle_dir: Path, int8_val_scores: Path, stage3_note: Optional[Dict] = None
) -> Path:
    """Rewrite a bundle's chain.json for the quantised engine, arm-B style.

    Stage 3 measured that INT8 under the frozen fp32 chain silently cut the
    review envelope by a third; refitting the staircase and floor on the INT8
    model's own validation scores restored it and improved Brier. This writes
    that refit chain: same validation images, same fitting code, scores from
    the INT8 graph.
    """
    import hashlib

    import numpy as np

    from vinspect.eval.calibration import IsotonicCalibrator

    bundle_dir = Path(bundle_dir)
    chain_path = bundle_dir / "chain.json"
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    rows = json.loads(Path(int8_val_scores).read_text(encoding="utf-8"))

    calibrator = IsotonicCalibrator().fit(
        [row["defect_score"] for row in rows],
        [row["label"] for row in rows],
    )
    clean_weak = [r["weak_evidence_px"] for r in rows if r["label"] == 0]
    floor = float(np.percentile(clean_weak, 95))

    int8_path = bundle_dir / "model.int8.onnx"
    chain["model"]["file"] = "model.int8.onnx"
    chain["model"]["engine"] = "onnxruntime-int8"
    chain["model"]["weights_sha256"] = hashlib.sha256(
        int8_path.read_bytes()
    ).hexdigest()
    chain["chain"]["calibrator"] = calibrator.summary()
    chain["chain"]["weak_floor"] = floor
    chain["provenance"]["quantisation"] = {
        "weights": "int8, per-channel, conv ops only",
        "activation_calibration": "stratified training images, stochastic graph",
        "chain_refit_on": "int8 validation scores (stage-3 arm B)",
        **({"stage3": stage3_note} if stage3_note else {}),
    }
    chain_path.write_text(json.dumps(chain, indent=2), encoding="utf-8")
    return chain_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export serving bundles.")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "carpet", "hazelnut"]
    )
    parser.add_argument("--out", type=Path, default=Path("bundles"))
    args = parser.parse_args(argv)

    for category in args.categories:
        destination = export_category(args.runs, category, args.out)
        chain = json.loads((destination / "chain.json").read_text())
        print(
            f"{category}: exported to {destination} "
            f"(floor {chain['chain']['weak_floor']:.0f} px, "
            f"{chain['chain']['calibrator']['n_steps']} calibrator steps, "
            f"split {chain['provenance']['split_digest'][:12]}...)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
