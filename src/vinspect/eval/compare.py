"""Re-evaluate trained checkpoints and put an interval on the split delta.

The matrix report gives point estimates. With 12 to 18 defective test images per
category, a point estimate on its own suggests a precision that is not there,
and the observed deltas swing in both directions -- which is what noise looks
like. This resamples the per-image scores so the question "is the gap larger
than the measurement error?" gets an answer rather than an opinion.

The two arms have different test sets, so their bootstrap distributions are
independent and the delta interval is taken from the difference of independent
draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from vinspect.data.mvtec import MVTecDataset, index_mvtec
from vinspect.data.splits import load_split, records_for_split
from vinspect.eval.metrics import SegmentationMetrics
from vinspect.models.unet import UNet
from torch.utils.data import DataLoader


def evaluate_checkpoint(
    checkpoint_path: Path, device: str = "cuda", split: str = "test"
) -> Tuple[SegmentationMetrics, Dict[str, object]]:
    """Load a checkpoint and re-score its own test split, keeping per-image rows."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = device if torch.cuda.is_available() else "cpu"

    model = UNet(
        base_channels=config["base_channels"],
        depth=config["depth"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    payload = load_split(Path(config["split_path"]))
    records = index_mvtec(Path(config["root"]), [config["category"]])
    rows = records_for_split(records, payload, split)

    loader = DataLoader(
        MVTecDataset(rows, image_size=config["image_size"]),
        batch_size=config["batch_size"],
        shuffle=False,
    )
    metrics = SegmentationMetrics(threshold=config["threshold"])
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            metrics.update(
                logits.cpu(), batch["mask"], batch["category"], batch["defect_type"]
            )
    return metrics, checkpoint


def delta_interval(
    grouped: SegmentationMetrics,
    random_run: SegmentationMetrics,
    category: str,
    metric: str = "iou",
    resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Interval on ``random - grouped`` from independent resamples of each arm."""

    def draws(metrics: SegmentationMetrics, offset: int) -> torch.Tensor:
        values = torch.tensor(
            [
                float(r[metric])
                for r in metrics.defective
                if r["category"] == category
            ]
        )
        generator = torch.Generator().manual_seed(seed + offset)
        index = torch.randint(
            len(values), (resamples, len(values)), generator=generator
        )
        return values[index].mean(dim=1)

    difference = draws(random_run, 1) - draws(grouped, 0)
    tail = (1.0 - confidence) / 2.0
    return {
        "point": float(difference.mean()),
        "low": float(difference.quantile(tail)),
        "high": float(difference.quantile(1.0 - tail)),
        "crosses_zero": bool(
            difference.quantile(tail) < 0 < difference.quantile(1.0 - tail)
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap intervals on the grouped-vs-random split delta."
    )
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "carpet", "hazelnut"]
    )
    parser.add_argument("--metric", default="iou", choices=("iou", "dice"))
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    header = (
        f"{'category':<10} {'n':>3}  {'grouped ' + args.metric:>22}  "
        f"{'random ' + args.metric:>22}  {'delta (random - grouped)':>26}"
    )
    print(header)
    print("-" * len(header))

    findings: Dict[str, Dict[str, object]] = {}
    for category in args.categories:
        arms = {}
        for kind in ("grouped", "random"):
            path = Path(args.runs) / f"{category}_{kind}" / "best.pt"
            if not path.is_file():
                print(f"{category:<10} missing {path}")
                break
            arms[kind], _ = evaluate_checkpoint(path, device=args.device)
        if len(arms) != 2:
            continue

        grouped = arms["grouped"].bootstrap(
            args.metric, category, resamples=args.resamples
        )
        randomised = arms["random"].bootstrap(
            args.metric, category, resamples=args.resamples
        )
        delta = delta_interval(
            arms["grouped"], arms["random"], category, args.metric, args.resamples
        )
        findings[category] = {
            "grouped": grouped,
            "random": randomised,
            "delta": delta,
        }

        print(
            f"{category:<10} {grouped['n']:>3}  "
            f"{grouped['point']:>6.3f} [{grouped['low']:.3f}, {grouped['high']:.3f}]  "
            f"{randomised['point']:>6.3f} [{randomised['low']:.3f}, {randomised['high']:.3f}]  "
            f"{delta['point']:>+7.3f} [{delta['low']:+.3f}, {delta['high']:+.3f}]"
            f"{'  n.s.' if delta['crosses_zero'] else '  *'}"
        )

    print("-" * len(header))
    print(
        "Intervals are 95% percentile bootstrap over defective test images.\n"
        "'n.s.' means the delta interval crosses zero: the measurement cannot\n"
        "distinguish it from no difference at this sample size."
    )

    out = Path(args.runs) / f"delta_{args.metric}.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
