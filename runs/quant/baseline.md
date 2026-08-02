# Quantisation experiment — stage 0 baseline

Recorded 2 August 2026, before any export or quantisation. Every later
before/after claim measures against this table.

## Hardware

| | local desktop | Cloud Run instance |
|---|---|---|
| CPU | AMD Ryzen 7 5800X (8C/16T, Zen 3) | Intel Xeon @ 2.80 GHz (Cascade Lake class) |
| AVX2 | yes | yes |
| AVX-512 | no | yes (f, bw, dq, cd, vl) |
| **VNNI (int8 dot-product)** | **no** | **yes (avx512_vnni)** |

Consequence, worth stating before measuring: INT8's full hardware fast path
exists **only on Cloud Run**. Local INT8 numbers will understate the deployed
speedup, so the deciding measurements are the deployed ones. (Cloud Run
instance draw could vary; the flags above are from one instance of the
deployed image, read via a one-off Cloud Run job.)

## Serving baseline (torch 2.8.0 fp32, the frozen 20-pass chain)

| metric | value |
|---|---|
| model | 2,159,937 params, fp32 checkpoint 8.3 MB per category |
| container image | 1.95 GB (torch CPU wheel dominates) |
| warm inspect, local (8 threads) | 3.46 s |
| warm inspect, Cloud Run (4 vCPU) | 8.0–8.9 s |
| cold start to first verdict, Cloud Run | 11.5 s |
| peak working set, local, one inspect | 829 MB (488 MB after model load) |
| verdict sanity | hazelnut hole/007 → fail p=0.9427 |

## Trust baseline (what compression must not silently break)

| metric | value |
|---|---|
| pooled test Brier | 0.019 (vs 0.15 base-rate guess) |
| pooled ECE | 0.026 |
| review rate (main pooled test, deployed routing) | 13.1% routed-first |
| holdout carpet/hole (frozen, clean test) | 17/17 caught, 0 silent, 4/61 clean routed |
| holdout hazelnut/cut (dev-flavoured) | 15/17 caught, 2 silent |
| image-level pooled | acc 97.9%, F1 0.941 (44 defective test images) |

## Measurement protocol for every variant

Same machine per comparison, same images, same 20-pass chain. Grid: warm/cold
latency (local and deployed), peak RSS, artifact and image size, max |Δ| vs
torch on deterministic maps, Brier/ECE, review rate, both holdout verdict
tables. Two calibration arms for any quantised variant: (A) frozen fp32-fitted
calibrator — the naive deployment, measures drift; (B) calibrator refit on the
variant's validation scores — measures whether recalibration repairs it.
