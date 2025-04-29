# tests/test_inference_demo.py

import joblib
import pandas as pd
import pytest
from pathlib import Path

from src.inference import load_pipeline, load_selected_features, demo_prediction

class DummyPipeline:
    """
    Stand‐in pipeline for testing: 
    - predict returns a constant label, 
    - predict_proba returns fixed probabilities.
    """
    def predict(self, X):
        # return one label per row
        return [99 for _ in range(len(X))]
    def predict_proba(self, X):
        # return a 2‐class probability for each row
        return [[0.1, 0.9] for _ in range(len(X))]

def test_load_selected_features(tmp_path):
    # Create a temporary features file
    feats_file = tmp_path / "features.txt"
    feats_file.write_text("f1\nf2\nf3\n")
    
    # Load it
    feats = load_selected_features(str(feats_file))
    
    # Assert correct type & contents
    assert isinstance(feats, list)
    assert feats == ["f1", "f2", "f3"]

def test_load_pipeline(tmp_path):
    # Dump our DummyPipeline to a temp .pkl
    pipeline_obj = DummyPipeline()
    pkl_path = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline_obj, str(pkl_path))
    
    # Load it back
    loaded = load_pipeline(str(pkl_path))
    
    # Should be an instance of DummyPipeline
    assert isinstance(loaded, DummyPipeline)

def test_demo_prediction_output(capsys):
    # Use our DummyPipeline and a small feature list
    pipe = DummyPipeline()
    features = ["x", "y", "z"]
    
    # Run demo_prediction, capture stdout
    demo_prediction(pipe, features)
    captured = capsys.readouterr()
    out = captured.out
    
    # Check that it prints the expected labels & phrases
    assert "Demo sample prediction:" in out
    assert "Demo sample probability of success:" in out
    
    # Since DummyPipeline.predict returns 99s, it should appear in output
    assert "99" in out