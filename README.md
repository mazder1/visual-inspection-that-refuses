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

Modules 01 and 02 are built, tested and measured. Modules 03 (abstention and
calibration) and 04 (the service) are not started.

## Results

Three categories, U-Net from scratch, 2.16M parameters, 512×512, trained on the
leakage-aware split. Per-pixel scores on the held-out test set, over defective
images only — on a clean image an overlap metric is degenerate and rewards
predicting nothing.

| category | defective test images | IoU | Dice |
|---|---:|---:|---:|
| bottle | 12 | 0.503 | 0.630 |
| carpet | 18 | 0.546 | 0.698 |
| hazelnut | 14 | 0.795 | 0.882 |

### The split delta — the finding

Same architecture, same seed, same hyperparameters; the only difference is
which split file was read. 95% percentile bootstrap over defective test images,
because a point estimate on 12 to 18 images implies a precision that is not
there.

| category | components appearing more than once | delta IoU (random − grouped) | |
|---|---:|---|---|
| bottle | 6 of 286 (2%) | −0.113 [−0.311, +0.096] | not distinguishable from zero |
| hazelnut | 72 of 423 (17%) | −0.027 [−0.118, +0.061] | not distinguishable from zero |
| **carpet** | **97 of 263 (37%)** | **+0.125 [+0.055, +0.200]** | **excludes zero** |

**The inflation appears where the leakage is.** The delta orders exactly with
how much component reuse each category actually contains. Carpet, where more
than a third of components appear in several images, is inflated by 0.125 IoU
by a random split. Bottle, with 2% reuse, shows nothing measurable, and its
negative point estimate is a small test set moving around.

Reported as three numbers rather than one average on purpose. A single
cross-category mean would have come to roughly zero and concluded, wrongly, that
the split does not matter.

### What this costs, stated plainly

**Per-defect-type coverage is incomplete on the grouped split.** Keeping
components together squeezed one defect class out of the test set for two of
three categories — hazelnut's grouped test set covers `cut`, `hole` and `print`
but not `crack`. That is a real gap, and preferable to closing it with a leaky
split.

**Clean parts are not clean.** On hazelnut, 40.7% of defect-free test images
carry at least one false-positive pixel, though the area is tiny (0.02% of
pixels). Nearly half of good parts would trigger something. That is the baseline
module 03's abstention layer has to improve on, and it is the number that
decides whether an operator keeps trusting the system.

### Reproducing

```sh
python -m vinspect.train.run_all --root data/mvtec_ad --out runs   # ~77 min on an RTX 3070
python -m vinspect.eval.compare --runs runs --metric iou
```

```sh
python -m vinspect.data.mvtec  --root data/mvtec_ad --check-load   # inventory
python -m vinspect.data.grouping --root data/mvtec_ad --cache scratch/scores.json
python -m vinspect.data.splits --root data/mvtec_ad --categories bottle hazelnut carpet
```

## What the split generator found

MVTec AD ships **no physical-instance IDs**. Filenames are just `000.png`,
`001.png` within each defect type, so "do not let two views of the same
component straddle the split" cannot be read off the metadata. Identity has to
be recovered from the pixels, and how you do that turns out to matter a lot.

**Perceptual hashing does not work here, and the reason is structural rather
than a tuning problem.** Every image in an MVTec category is the same object
type, centred, in a fixed rig, under fixed lighting. The coarse layout a
difference hash encodes is therefore constant across the whole category by
construction, while what distinguishes two physical parts is fine surface
texture — exactly what the hash is built to discard. Sweeping it shows no
stable operating point: cluster size jumps from 1 straight into the hundreds
with no plateau, because the similarity graph is nearly complete and
single-linkage chains across it. The method is kept in the repo as a measured
negative control rather than deleted.

**Keypoint matching does work.** ORB features, a Lowe ratio test, then RANSAC
under a partial-affine model — rotation, translation and uniform scale, which
is exactly the transform a part undergoes when lifted and set back down on the
rig. Genuine repeats separate cleanly from chance.

And MVTec AD **does** reuse physical components: the same hazelnut appears in
multiple images at different rotations, identifiable by matching shell
striations. A random split would put those on both sides of the boundary.

**Single linkage chains, and it matters.** Connected components over the match
graph merges A and C whenever both match B, even when A and C do not match each
other at all. That produced a 22-image hazelnut "component" of visibly
different nuts, held together by cracked shells sharing exposed-kernel texture
pairwise. Complete linkage — requiring the cluster to be a clique above the
threshold — is the shape the claim actually has, since *all* images of one part
should match each other, and it removes the chaining entirely.

**Thresholds are calibrated per category against that category's own chance
level**, at twice its 99th-percentile pair score with a floor. Per-category is
necessary rather than fussy: on the three categories measured, the 99th
percentile of pair scores ranges from 4 to 18, because how much structure two
*different* parts of the same type share is a property of the part. A size cap
is retained as an assertion — if it ever binds, the matcher is not separating on
that category and the split is refused rather than built.

Cluster contact sheets are written for visual sign-off. A threshold nobody has
looked at is not a justified threshold, and every rejected variant above was
rejected by looking at what it merged.

## The split, as generated

Three categories, 1,190 images, at 60/20/20. Committed under `splits/` with a
content hash, so a result can be tied to the split it was measured on.

| | train | val | test |
|---|---:|---:|---:|
| images | 714 (60.0%) | 239 (20.1%) | 237 (19.9%) |
| defective | 133 | 45 | 44 |

Calibrated thresholds: `bottle` 36, `hazelnut` 22, `carpet` 20 — the spread is
the point, and it comes from each category's own chance level.

Grouping recovered **972 components from 1,190 images**. 393 images (33%) sit
in a group with at least one other image; the largest group has 5 members, and
the size distribution is `{1: 797, 2: 141, 3: 27, 4: 5, 5: 2}`.

**The headline number:**

| split | components straddling a boundary |
|---|---:|
| grouped | **0** of 972 |
| random | **107** of 972 |

So a random split — stratified by category and label, i.e. the strongest
reasonable version of the naive approach, not a strawman — puts 107 physical
components on both sides of the boundary. Whatever score that split reports is
partly a score for recognising parts the model has already seen. How much it
inflates by is the module 02 measurement; that it inflates at all is now
established rather than assumed.

One honest cost of grouping: in `bottle`, val and test each cover 2 of the 3
defect types under the grouped split, against 3 under the random one. With 63
defective bottles across 3 types, the group constraint is tight enough to drop
a type from the smaller splits. Per-defect-type reporting on `bottle` will have
a gap, and that is preferable to closing it with a leaky split.

## The segmenter, as specified

U-Net implemented from the paper. Every choice below was made deliberately and
is enforced by a test in `tests/test_unet.py`, so none of them can drift.

| decision | choice | why |
|---|---|---|
| Upsampling | bilinear + conv | transposed convolution's uneven kernel overlap produces checkerboard artifacts, which land directly on mask boundaries |
| Skip connections | concatenate | encoder detail and decoder context are different kinds of information; concatenation lets the next conv learn the mixing rather than pre-committing to a sum |
| Convolutions | **padded** | *deviates from the paper* — see below |
| Normalisation | GroupNorm, 8 groups | statistics independent of batch size, and identical in train and eval; the service handles one image at a time |
| Width / depth | base 16, depth 4 | ~2.2M parameters against 133 defective images; base 64 would be ~34M |
| Dropout | decoder, p≈0.1 | MC dropout in module 03 needs it present in the trained weights |
| Loss | Focal(γ=2) + Tversky(α=0.3, β=0.7) | focal decides which pixels get gradient, Tversky what the region should look like and which way to err |
| Resolution | 512×512 | carpet defects average 1.67% of pixels; thin threads do not survive 256 |
| Models | three, one per category | the categories are visually unrelated; a shared model spends capacity telling them apart |

### The one deviation from the paper

The 2015 paper uses **unpadded** convolutions, so a 572×572 input yields a
388×388 output and full images need the overlap-tile strategy. That was a
workaround for the GPU memory of the time.

This implementation pads, so output size equals input size and the predicted
mask aligns with the ground truth directly, with no cropping step in the
evaluation path. Nothing is gained by reproducing a memory workaround on
hardware that does not need it, and the cropping would be one more place for an
off-by-one to hide.

### On the loss

Both alternatives run from configuration rather than a code change, because the
region term should earn its place by measurement: `tversky_weight=0` gives focal
alone — which is what DRAEM uses for its own U-Net-shaped segmentation head, at
near-SOTA on this dataset — and `alpha=beta=0.5` gives plain Dice.

## Data

MVTec AD, 5,354 images, of which **1,258 are defective**. That second number is
the one that constrains everything: recall, calibration and the risk-coverage
curve are all measured on defective images. See [DATA.md](DATA.md) for
provenance, licence and the full per-category inventory.

The splits cover three categories rather than fifteen — one texture and two
object — so that each split holds enough defective images to calibrate on.

## Data and licence

Primary dataset is MVTec AD, which is licensed for non-commercial research use. The licence,
version and source will be recorded in this repo and honoured.
