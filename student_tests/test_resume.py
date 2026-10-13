import json

import pytest

from solution.experiment import read_predictions
from solution.training import batch_indices


def test_resume_stream_is_identical_at_epoch_boundary():
    full, end = batch_indices(7, 19, 0, 42)
    first, cursor = batch_indices(7, 8, 0, 42)
    second, resumed = batch_indices(7, 11, cursor, 42)
    assert first+second == full and resumed == end


def test_trailing_partial_prediction_is_removed_before_resume(tmp_path):
    path = tmp_path/"pred.jsonl"
    good = json.dumps({"image_id": "a", "raw": "[]"})+"\n"
    path.write_text(good+'{"image_id":', encoding="utf-8")
    assert read_predictions(path) == {"a": "[]"}
    assert path.read_text() == good


def test_intact_last_line_gets_newline(tmp_path):
    path = tmp_path/"pred.jsonl"
    path.write_text(json.dumps({"image_id": "a", "raw": "[]"}), encoding="utf-8")
    assert read_predictions(path) == {"a": "[]"}
    assert path.read_bytes().endswith(b"\n")


def test_corruption_in_middle_is_not_hidden(tmp_path):
    path = tmp_path/"pred.jsonl"
    path.write_text('bad\n'+json.dumps({"image_id": "a", "raw": "[]"}), encoding="utf-8")
    with pytest.raises(ValueError, match="before final"):
        read_predictions(path)
