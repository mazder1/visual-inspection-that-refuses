"""Engine benchmark, meant to run ON the deployment hardware as a Cloud Run job.

Local numbers cannot answer the stage-4 question: the desktop Ryzen has no
VNNI, so INT8 loses there by construction. This prints the same 20-pass
workload for torch fp32, ONNX fp32 and ONNX INT8 on whatever machine it runs
on, with several repetitions because Cloud Run instance draws vary, plus the
CPU flags so the result is attributable to hardware rather than luck.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

BUNDLE = Path("bundles/hazelnut")
PASSES = 20
REPS = 3


def cpu_summary() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        model = next(
            (line.split(":", 1)[1].strip() for line in text.splitlines()
             if line.startswith("model name")),
            "unknown",
        )
        flags = next(
            (line for line in text.splitlines() if line.startswith("flags")), ""
        )
        interesting = [f for f in ("avx2", "avx512f", "avx512_vnni", "avx_vnni")
                       if f" {f} " in flags + " "]
        return f"{model} | {' '.join(interesting) or 'no avx flags found'}"
    except OSError:
        return "cpuinfo unavailable (not linux)"


def main() -> int:
    print(f"cpu: {cpu_summary()}")
    print(f"torch threads: {torch.get_num_threads()}")

    image = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(0))
    array = image.numpy()

    from vinspect.serve.onnx_export import load_bundle_model, make_session
    from vinspect.uncertainty.mc_dropout import enable_dropout

    model, _ = load_bundle_model(BUNDLE)
    enable_dropout(model)

    def bench(label, fn):
        fn()  # warm-up
        for rep in range(REPS):
            started = time.perf_counter()
            for _ in range(PASSES):
                fn()
            print(f"  {label:<12} rep {rep + 1}: {time.perf_counter() - started:.2f} s")

    with torch.no_grad():
        bench("torch fp32", lambda: model(image))

    for engine, filename in (("onnx fp32", "model.onnx"), ("onnx int8", "model.int8.onnx")):
        path = BUNDLE / filename
        if not path.is_file():
            print(f"  {engine:<12} SKIPPED: {path} not in image")
            continue
        session = make_session(path)
        bench(engine, lambda: session.run(None, {"image": array}))

    # Keep numpy referenced so the import is unambiguous under linters.
    _ = np.zeros(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
