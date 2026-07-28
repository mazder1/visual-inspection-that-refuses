# Visual Inspection That Refuses

An AI defect-segmentation system for manufactured parts that knows when it does not know.

Given a photograph of a part, it returns a per-pixel defect mask, an image-level verdict of
**defective**, **clean**, or **no-call**, and a calibrated confidence. Parts the model is
confident about are decided automatically. Parts it is unsure about are routed to a human.

## Why

A segmentation model reporting 94% IoU tells a plant manager nothing useful. The two numbers
a production line actually runs on are *how many defects escape* and *at what human review
load*. This project reports those, as a risk-coverage curve, instead of a single headline
score.

The cost structure is asymmetric: a false alarm costs a technician thirty seconds, a missed
defect ships a broken part into an assembly. A confident wrong pass is worse than an
abstention, so abstention is designed in from the first commit.

## What it is made of

- **Honest splits.** Grouped by physical object instance so near-duplicate views cannot
  straddle the split. The same model is also trained on a random split, and both numbers are
  reported side by side.
- **U-Net from the paper.** Implemented in PyTorch from the original architecture, not from a
  segmentation library.
- **An abstention layer.** Per-pixel uncertainty aggregated to an image-level confidence,
  calibrated with isotonic regression, swept into a risk-coverage curve.
- **A deployed service.** Containerised, on a live public URL, with measured cold-start and
  warm latency.

## Scope

In scope: a single still image of one part from a supported category, arriving at an HTTP
endpoint. Out of scope: image collection, camera and lighting rig design, line integration,
PLC or robot control, and root-cause analysis.

This is decision support. It routes parts to an inspector and never makes a final accept or
reject decision on its own. It is not qualified for production use.

## Status

Early. Nothing measured yet.

## Data and licence

Primary dataset is MVTec AD, which is licensed for non-commercial research use. The licence,
version and source will be recorded in this repo and honoured.
