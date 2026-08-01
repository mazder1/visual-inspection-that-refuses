"""Run the full matrix: every category, on both splits.

This produces module 02's headline. Six runs, identical in every respect except
which split file they read, so the difference between them is attributable to
leakage rather than to anything else.

One thing to read carefully in the output. The grouped and random runs are
**not scored on the same images** -- each split has its own test set. That is not
a flaw in the comparison, it is the comparison: the random split's test set
contains components the model already saw in training, and the question is how
much that inflates the number. A reader who assumes the two columns are
measured on the same data will draw the wrong conclusion, so the report says so
explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vinspect.train.loop import TrainConfig, train


def run_matrix(
    root: Path,
    split_paths: Sequence[Path],
    categories: Sequence[str],
    out_dir: Path,
    **overrides: object,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    total = len(categories) * len(split_paths)
    started = time.time()

    for index, category in enumerate(categories):
        for offset, split_path in enumerate(split_paths):
            step = index * len(split_paths) + offset + 1
            print(f"\n{'=' * 72}")
            print(f"[{step}/{total}] {category} on {Path(split_path).name}")
            print(f"{'=' * 72}")
            config = TrainConfig(
                root=Path(root),
                split_path=Path(split_path),
                category=category,
                out_dir=Path(out_dir),
                **overrides,  # type: ignore[arg-type]
            )
            results.append(train(config))
            elapsed = (time.time() - started) / 60
            print(f"\nelapsed {elapsed:.1f} min after {step} of {total} runs")

    return results


def format_comparison(results: Sequence[Dict[str, object]]) -> str:
    """The split delta, per category. The gap is the finding."""
    by_key = {(r["category"], r["split_kind"]): r for r in results}
    categories = sorted({str(r["category"]) for r in results})

    header = (
        f"{'category':<12} {'grouped IoU':>12} {'random IoU':>11} {'delta':>8}   "
        f"{'grouped Dice':>13} {'random Dice':>12} {'delta':>8}"
    )
    lines = [header, "-" * len(header)]

    for category in categories:
        grouped = by_key.get((category, "grouped"))
        random_run = by_key.get((category, "random"))
        if not grouped or not random_run:
            lines.append(f"{category:<12}  incomplete pair, skipped")
            continue

        g = grouped["test"]["by_category"][category]  # type: ignore[index]
        r = random_run["test"]["by_category"][category]  # type: ignore[index]
        lines.append(
            f"{category:<12} {g['iou']:>12.3f} {r['iou']:>11.3f} "
            f"{r['iou'] - g['iou']:>+8.3f}   "
            f"{g['dice']:>13.3f} {r['dice']:>12.3f} "
            f"{r['dice'] - g['dice']:>+8.3f}"
        )

    lines.append("-" * len(header))
    lines.append(
        "A positive delta means the random split scored higher. The two columns "
        "are not\nmeasured on the same images: each split has its own test set, "
        "and the random one\ncontains components the model saw in training. That "
        "gap is what a naive\nevaluation would have reported as model quality."
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train every category on both splits.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        type=Path,
        nargs="+",
        default=[Path("splits/grouped.json"), Path("splits/random.json")],
    )
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "carpet", "hazelnut"]
    )
    parser.add_argument("--out", dest="out_dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tversky-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)

    results = run_matrix(
        args.root,
        args.splits,
        args.categories,
        args.out_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        tversky_weight=args.tversky_weight,
        alpha=args.alpha,
        beta=args.beta,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("THE SPLIT DELTA")
    print(f"{'=' * 72}")
    print(format_comparison(results))
    print(f"\nwritten to {out_dir / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
