import pytest

from solution.check_quality import checks
from solution.reporting import paired_delta


def test_paired_session_interval_uses_groups_and_direction():
    rows = [{"image_id": str(i), "session": f"s{i//3}", "objects": [
        {"label": "pothole", "yolo": [.5, .5, .2, .2]}], "width": 640, "height": 360} for i in range(12)]
    bad = {e["image_id"]: "[]" for e in rows}
    good = {e["image_id"]: '[{"label":"pothole","bbox_2d":[400,400,600,600]}]' for e in rows}
    forward = paired_delta(rows, bad, good, n_boot=100)
    reverse = paired_delta(rows, good, bad, n_boot=100)
    assert forward["n_sessions"] == 4
    assert forward["delta_ci95"] == pytest.approx([1, 1])
    assert reverse["delta_ci95"] == pytest.approx([-1, -1])


def test_quality_checker_rejects_unrun_template(tmp_path):
    result = checks(tmp_path)
    assert result and any(not ok for ok, _ in result)
    assert any("e5" in message for ok, message in result if not ok)
