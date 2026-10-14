import pytest
from PIL import Image

from solution.check_quality import checks
from solution.reporting import paired_delta
from solution.reporting import build_report
from solution.storage import write_json, append_jsonl
from scripts.score import evaluate


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


def test_report_is_built_from_fixture_evidence_with_six_images(tmp_path, monkeypatch):
    """Synthetic fixtures stay in pytest temp directories, never submission evidence."""
    monkeypatch.chdir(tmp_path)
    rows = []
    for i in range(6):
        image = tmp_path/f"data/images/{i}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), "gray").save(image)
        rows.append({"image_id": str(i), "session": f"session-{i//2}", "image_path": f"images/{i}.jpg",
                     "width": 640, "height": 360, "objects": [
                         {"label": c, "yolo": [.2+j*.25, .5, .1, .1]} for j, c in enumerate(("pothole", "crack", "manhole"))]})
    from solution.core import objects_to_answer
    predictions = {e["image_id"]: objects_to_answer(e["objects"]) if int(e["image_id"])%2 else "[]" for e in rows}
    for e in rows:
        append_jsonl("data/splits/test.jsonl", e)
    write_json("data/splits/meta.json", {"dataset_version": "test-fixture-only", "counts": {
        "test": {"images": 6, "boxes": {"pothole": 6, "crack": 6, "manhole": 6}}}})
    names = ("e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0", "e5")
    for name in names:
        result = evaluate(rows, predictions, n_boot=20)
        result.update(name=name.upper(), meta={"trainable_params": 10, "peak_vram_gb": 1, "train_minutes": 1})
        write_json(f"report/evidence/{name}.json", result)
        for image_id, raw in predictions.items():
            append_jsonl(f"work/{name}_test.jsonl", {"image_id": image_id, "raw": raw})
        append_jsonl(f"work/{name}/train_log.jsonl", {"step": 1, "loss": 1})
    write_json("report/selection.json", {"selected": "E1", "validation_f1": {"E1": .5}})
    write_json("report/runs/e5.json", {"trainable_params": 10,
                                     "resume_from_checkpoint": "fixture/checkpoint-5",
                                     "metrics": {"train_loss": .1}})
    write_json("report/validation/e5.json", {"micro": {"f1": .5}, "parse_valid_rate": 1})
    write_json("report/configs/e1.json", {"coord": "norm1000"})
    write_json("work/e1/run.json", {"architecture": {"vit_without_projector": 1, "projector": 2,
                                                     "llm_and_head": 3, "total": 6},
                                   "encoding_audit": {"vision_tokens": 220, "assistant_tokens": 20,
                                                      "supervised_text": "[]<|im_end|>"}})
    report = build_report(author="Test Fixture")
    content = report.read_text(encoding="utf-8")
    assert "Test Fixture" in content and "test-fixture-only" in content and "<điền" not in content
    assert len(list((tmp_path/"report/figures").glob("prediction_*.png"))) == 6
    assert (tmp_path/"report/figures/learning_and_f1.png").exists()
    assert "CI chứa 0" in content  # Identical predictions must not imply a proven gain.
