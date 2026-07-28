# Data provenance

## MVTec AD

| | |
|---|---|
| Source | <https://www.mvtec.com/company/research/datasets/mvtec-ad> |
| Archive | `mvtec_anomaly_detection.tar.xz` |
| Size | 5,264,982,680 bytes (4.90 GiB) |
| Retrieved | 28 July 2026 |
| Licence | CC BY-NC-SA 4.0 — **non-commercial research use only** |

Not a Kaggle or Hugging Face mirror. Downloaded from the download page linked
above, which is MVTec's own distribution.

### Citation

Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
*MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly
Detection.* CVPR 2019.

Extended version: *The MVTec Anomaly Detection Dataset: A Comprehensive
Real-World Dataset for Unsupervised Anomaly Detection.* IJCV 2021.

### What the licence means for this repo

BY-NC-SA. Attribution is given above, use here is non-commercial research, and
no part of the dataset is redistributed — `data/` is gitignored and only the
split membership files, which contain relative keys and no pixels, are
committed. Any derivative that redistributes images inherits SA terms.

This repo is not qualified for production use and makes no commercial claim.

## Measured inventory

Produced by `python -m vinspect.data.mvtec --root data/mvtec_ad`, and matching
the published figures exactly.

| | count |
|---|---|
| Images | 5,354 |
| Clean | 4,096 |
| Defective | 1,258 |
| Categories | 15 (5 texture, 10 object) |
| Defect types | 73 |
| Bytes on disk | 5,267 MB |

Per category:

| category | kind | total | clean | defect | types |
|---|---|---:|---:|---:|---:|
| bottle | object | 292 | 229 | 63 | 3 |
| cable | object | 374 | 282 | 92 | 8 |
| capsule | object | 351 | 242 | 109 | 5 |
| carpet | texture | 397 | 308 | 89 | 5 |
| grid | texture | 342 | 285 | 57 | 5 |
| hazelnut | object | 501 | 431 | 70 | 4 |
| leather | texture | 369 | 277 | 92 | 5 |
| metal_nut | object | 335 | 242 | 93 | 4 |
| pill | object | 434 | 293 | 141 | 7 |
| screw | object | 480 | 361 | 119 | 5 |
| tile | texture | 347 | 263 | 84 | 5 |
| toothbrush | object | 102 | 72 | 30 | 1 |
| transistor | object | 313 | 273 | 40 | 4 |
| wood | texture | 326 | 266 | 60 | 5 |
| zipper | object | 391 | 272 | 119 | 7 |

**1,258 is the number that constrains the split, not 5,354.** Recall,
calibration and the risk-coverage curve are all measured on defective images.

Defect area varies by three orders of magnitude across categories — from 10.9%
of pixels on `bottle/broken_large` down to 0.06% on `pill/color`. Any per-pixel
loss or metric averaged across categories is dominated by that spread rather
than by model quality, which is one more reason the brief reports per category.

## Reproducing

```sh
curl -L -C - -o data/mvtec_anomaly_detection.tar.xz \
  "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
mkdir -p data/mvtec_ad
tar -xJf data/mvtec_anomaly_detection.tar.xz -C data/mvtec_ad
```

MVTec has rotated this share link before; an older one now 404s. If it fails,
take the current link from the download page rather than a mirror.
