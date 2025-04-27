# tests/test_smoke.py

import pytest

def test_pipeline_module_importable():
    """
    Smoke-test that src.pipeline can be imported and has a 'main' function.
    """
    import src.pipeline
    assert hasattr(src.pipeline, "main"), "pipeline.main must exist"

def test_inference_app_startup():
    """
    Smoke-test that the FastAPI app can be imported and exposes the app object.
    """
    from fastapi.testclient import TestClient
    import src.inference

    client = TestClient(src.inference.app)
    # calling without payload should return 422 (validation error)
    resp = client.post("/predict", json={})
    assert resp.status_code == 400