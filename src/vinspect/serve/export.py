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
from typing import List, Optional

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
