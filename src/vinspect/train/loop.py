"""Train one model, on one category, against one split.

Six runs make the module 02 result: three categories times the grouped and
random splits. The gap between the two is the finding module 01 was built to
measure.

Two choices here are load-bearing.

**Model selection uses Dice over defective validation images only.** On a clean
image the target is empty, and any overlap metric on two empty sets is
degenerate -- a model predicting nothing scores a perfect 1.0. Selecting on a
metric that includes clean images would crown the model that does nothing.
False-alarm area on clean images is logged every epoch alongside it, because
selecting on Dice alone is blind to whether recall was bought with false alarms.

**The checkpoint records the split's SHA-256.** That is the point of module 01:
a number should be permanently attached to the split it was measured on, so it
can never be quietly reattributed to the other one.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vinspect.data.mvtec import MVTecDataset, MVTecRecord, index_mvtec
from vinspect.data.splits import SPLITS, digest, load_split, records_for_split
from vinspect.eval.metrics import SegmentationMetrics
from vinspect.models.unet import UNet
from vinspect.train.augment import AugmentedDataset, DefectAugmentation
from vinspect.train.loss import FocalTverskyLoss
from vinspect.train.sampler import StratifiedBatchSampler


@dataclass
class TrainConfig:
    root: Path
    split_path: Path
    category: str
    out_dir: Path = Path("runs")
    image_size: int = 512
    batch_size: int = 8
    epochs: int = 120
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    base_channels: int = 16
    depth: int = 4
    dropout: float = 0.1
    gamma: float = 2.0
    alpha: float = 0.3
    beta: float = 0.7
    tversky_weight: float = 1.0
    threshold: float = 0.5
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _positive_rate(dataset: MVTecDataset) -> float:
    """Measured defect-pixel fraction, used to initialise the output bias.

    Focal loss can start with vanishing gradients if the untrained model sits at
    0.5 while the base rate is under 1%. Measured per category rather than
    assumed, because the rate ranges from 0.37% on carpet to 1.87% on bottle.
    """
    total = torch.tensor(0.0)
    for i in range(len(dataset)):
        total += dataset[i]["mask"].mean()
    return float(total / max(len(dataset), 1))


def build_loaders(
    config: TrainConfig,
) -> Tuple[Dict[str, DataLoader], Dict[str, List[MVTecRecord]], StratifiedBatchSampler, float]:
    payload = load_split(config.split_path)
    if config.category not in payload["categories"]:  # type: ignore[operator]
        raise ValueError(
            f"{config.category!r} is not in this split; it covers "
            f"{payload['categories']}"
        )

    records = index_mvtec(config.root, [config.category])
    per_split = {s: records_for_split(records, payload, s) for s in SPLITS}

    train_base = MVTecDataset(per_split["train"], image_size=config.image_size)
    prior = _positive_rate(train_base)

    augmented = AugmentedDataset(
        train_base, DefectAugmentation(config.category, enabled=True)
    )
    sampler = StratifiedBatchSampler(
        [r.label for r in per_split["train"]],
        batch_size=config.batch_size,
        seed=config.seed,
    )

    # persistent_workers matters a lot here. Without it, DataLoader tears down
    # and respawns its workers every epoch, and Windows spawns processes rather
    # than forking -- each respawn re-imports torch in four processes. On an
    # epoch that is otherwise about two seconds of work, that overhead dominates
    # everything else.
    persistent = config.num_workers > 0
    loaders = {
        "train": DataLoader(
            augmented,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=persistent,
        )
    }
    for split in ("val", "test"):
        loaders[split] = DataLoader(
            MVTecDataset(per_split[split], image_size=config.image_size),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=persistent,
        )
    return loaders, per_split, sampler, prior


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: str, threshold: float
) -> SegmentationMetrics:
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        metrics.update(
            logits.cpu(), batch["mask"], batch["category"], batch["defect_type"]
        )
    return metrics


def train(config: TrainConfig, verbose: bool = True) -> Dict[str, object]:
    set_seed(config.seed)
    device = config.device if torch.cuda.is_available() else "cpu"
    use_amp = config.amp and device == "cuda"

    loaders, per_split, sampler, prior = build_loaders(config)
    payload = load_split(config.split_path)
    split_digest = digest(payload)

    model = UNet(
        base_channels=config.base_channels,
        depth=config.depth,
        dropout=config.dropout,
        prior=prior if prior > 0 else None,
    ).to(device)
    criterion = FocalTverskyLoss(
        gamma=config.gamma,
        alpha=config.alpha,
        beta=config.beta,
        tversky_weight=config.tversky_weight,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

    run_dir = Path(config.out_dir) / f"{config.category}_{payload['kind']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n_defective_val = sum(r.label for r in per_split["val"])
    if verbose:
        print(
            f"{config.category}: {len(per_split['train'])} train "
            f"({sum(r.label for r in per_split['train'])} defective), "
            f"{len(per_split['val'])} val ({n_defective_val} defective), "
            f"{len(per_split['test'])} test"
        )
        print(
            f"measured defect-pixel rate {prior:.4%}, output bias initialised "
            f"to match; batches are {sampler.realised_rate:.1%} defective "
            f"against a natural {sampler.natural_rate:.1%}"
        )
        if n_defective_val < 10:
            print(
                f"WARNING: {n_defective_val} defective validation images. "
                f"Differences between checkpoints will not be meaningful."
            )

    history: List[Dict[str, float]] = []
    best_dice, best_epoch, since_improvement = -1.0, -1, 0
    started = time.time()

    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        totals = {"loss": 0.0, "focal": 0.0, "tversky": 0.0}
        steps = 0

        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
            # The loss casts to float32 internally: the region term sums 262k
            # values per image, which bfloat16 cannot accumulate accurately.
            loss, parts = criterion(logits, masks)
            loss.backward()
            optimiser.step()

            totals["loss"] += float(loss.detach())
            totals["focal"] += float(parts["focal"])
            totals["tversky"] += float(parts["tversky"])
            steps += 1

        schedule.step()
        metrics = evaluate(model, loaders["val"], device, config.threshold)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(steps, 1),
            "focal": totals["focal"] / max(steps, 1),
            "tversky": totals["tversky"] / max(steps, 1),
            "val_dice": metrics.mean_dice(),
            "val_false_alarm_area": metrics.false_alarm_area(),
            "learning_rate": schedule.get_last_lr()[0],
        }
        history.append(row)

        if row["val_dice"] > best_dice:
            best_dice, best_epoch, since_improvement = row["val_dice"], epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {**asdict(config), "root": str(config.root),
                               "split_path": str(config.split_path),
                               "out_dir": str(config.out_dir)},
                    "split_digest": split_digest,
                    "split_kind": payload["kind"],
                    "prior": prior,
                    "epoch": epoch,
                    "val_dice": best_dice,
                },
                run_dir / "best.pt",
            )
        else:
            since_improvement += 1

        last_epoch = epoch == config.epochs - 1 or since_improvement >= config.patience
        if verbose and (epoch % 5 == 0 or since_improvement == 0 or last_epoch):
            print(
                f"  epoch {epoch:>3}  loss {row['loss']:.4f} "
                f"(focal {row['focal']:.4f}, tversky {row['tversky']:.4f})  "
                f"val Dice {row['val_dice']:.4f}  "
                f"false-alarm {row['val_false_alarm_area']:.4%}"
                f"{'  *' if since_improvement == 0 else ''}"
            )

        if since_improvement >= config.patience:
            if verbose:
                print(f"  stopping at epoch {epoch}, no improvement for {config.patience}")
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test"], device, config.threshold)

    result = {
        "category": config.category,
        "split_kind": payload["kind"],
        "split_digest": split_digest,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "prior": prior,
        "batch_defective_rate": sampler.realised_rate,
        "natural_defective_rate": sampler.natural_rate,
        "minutes": (time.time() - started) / 60.0,
        "test": test_metrics.summary(),
        "history": history,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nbest epoch {best_epoch}, val Dice {best_dice:.4f}")
        print(test_metrics.format_report())
        print(f"\nsplit digest {split_digest}")
        print(f"written to {run_dir}")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train one U-Net on one category.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", dest="split_path", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, default=Path("runs"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument(
        "--tversky-weight",
        type=float,
        default=1.0,
        help="0 gives focal alone, the DRAEM configuration",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args(argv)

    train(TrainConfig(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
