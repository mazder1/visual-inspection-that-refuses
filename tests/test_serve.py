"""Tests for the service, against a tiny synthetic bundle.

The suite must not depend on trained checkpoints or the dataset, so the
fixture builds a real (small) UNet, saves it in checkpoint format, and writes
a hand-built chain.json. What is pinned: the response contract, the three
verdict paths, malformed-upload handling, the rate limit, and that the mask
round-trips as a PNG.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
import torch
from PIL import Image

from vinspect.models.unet import UNet

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundles")
    destination = root / "widget"
    destination.mkdir()

    model = UNet(base_channels=8, depth=2, dropout=0.1)
    torch.save({"model": model.state_dict()}, destination / "model.pt")

    chain = {
        "category": "widget",
        "model": {
            "base_channels": 8,
            "depth": 2,
            "dropout": 0.1,
            "image_size": 64,
            "weights_sha256": "test",
        },
        "chain": {
            "mc_passes": 4,
            "threshold": 0.5,
            "weak_threshold": 0.33,
            "weak_floor": 50.0,
            "no_call_band": [0.2, 0.8],
            "calibrator": {
                "n_steps": 2,
                "base_rate": 0.2,
                "pseudocount": 1.0,
                "steps": [
                    {"scores": [0.0, 10.0], "probability": 0.01, "count": 40, "raw_rate": 0.0},
                    {"scores": [100.0, 5000.0], "probability": 0.9, "count": 10, "raw_rate": 1.0},
                ],
            },
        },
        "provenance": {"split_digest": "test", "split_kind": "grouped"},
        "disclaimer": "test disclaimer",
    }
    (destination / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def client(bundle_dir, tmp_path_factory):
    import vinspect.serve.app as service

    service.BUNDLE_DIR = bundle_dir
    service._predictors.clear()
    service._requests.clear()
    return fastapi_testclient.TestClient(service.app)


def _png_bytes(size=64, value=128):
    image = Image.new("RGB", (size, size), (value, value, value))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_healthz_lists_bundles(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "widget" in body["bundles"]


def test_inspect_returns_the_contract(client):
    response = client.post(
        "/inspect/widget", files={"file": ("part.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()

    for key in (
        "verdict", "probability_defective", "probability_is_supported",
        "defect_score", "weak_evidence_px", "n_regions", "regions",
        "mask_png_base64", "latency_ms", "provenance", "disclaimer",
        "explanation", "mc_passes",
    ):
        assert key in body, f"response is missing {key}"
    assert body["verdict"] in ("pass", "fail", "no-call")
    assert 0.0 <= body["probability_defective"] <= 1.0
    assert body["mc_passes"] == 4


def test_mask_round_trips_as_png(client):
    response = client.post(
        "/inspect/widget", files={"file": ("part.png", _png_bytes(), "image/png")}
    )
    mask = Image.open(io.BytesIO(base64.b64decode(response.json()["mask_png_base64"])))
    assert mask.size == (64, 64)
    assert mask.mode == "L"


def test_unknown_category_is_a_404_naming_the_known_ones(client):
    response = client.post(
        "/inspect/gadget", files={"file": ("part.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 404
    assert "widget" in response.json()["detail"]


def test_malformed_upload_is_a_422_not_a_crash(client):
    response = client.post(
        "/inspect/widget",
        files={"file": ("part.png", b"this is not an image", "image/png")},
    )
    assert response.status_code == 422


def test_empty_upload_is_a_400(client):
    response = client.post(
        "/inspect/widget", files={"file": ("part.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_oversized_upload_is_a_413(client):
    import vinspect.serve.app as service

    payload = b"x" * (service.MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/inspect/widget", files={"file": ("part.png", payload, "image/png")}
    )
    assert response.status_code == 413


def test_rate_limit_kicks_in(client):
    import vinspect.serve.app as service

    service._requests.clear()
    original = service.RATE_LIMIT
    service.RATE_LIMIT = 3
    try:
        payload = {"file": ("part.png", _png_bytes(), "image/png")}
        for _ in range(3):
            assert client.post("/inspect/widget", files=payload).status_code == 200
        assert client.post("/inspect/widget", files=payload).status_code == 429
    finally:
        service.RATE_LIMIT = original
        service._requests.clear()


def test_index_page_serves(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "no-call" in response.text


def test_verdict_paths_of_the_step_calibrator(bundle_dir):
    from vinspect.serve.predictor import StepCalibrator

    chain = json.loads((bundle_dir / "widget" / "chain.json").read_text())
    calibrator = StepCalibrator(chain["chain"]["calibrator"])

    assert calibrator.predict(5.0) == pytest.approx(0.01)
    assert calibrator.predict(500.0) == pytest.approx(0.9)
    assert calibrator.supported(5.0)
    assert not calibrator.supported(50.0), "the 10..100 gap must be unsupported"
    assert not calibrator.supported(9999.0)
