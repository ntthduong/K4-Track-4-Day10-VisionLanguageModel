from solution.storage import read_json
from solution.worker import execute


def test_worker_keeps_training_state_and_checkpoint_path(tmp_path, monkeypatch):
    import solution.training
    monkeypatch.setattr(solution.training, "train_sft", lambda cfg, data, work: (
        tmp_path/"checkpoint-300", {"step": 300, "train_minutes": 12, "config": cfg}))
    output = tmp_path/"result.json"
    execute({"operation": "train", "config": {"name": "test-fixture"}, "data_dir": "fixture-data",
             "work_dir": "fixture-work", "result_path": str(output)})
    saved = read_json(output)
    assert saved["state"]["step"] == 300
    assert saved["checkpoint"] == str(tmp_path/"checkpoint-300")


def test_worker_inference_routes_exact_split_and_checkpoint(tmp_path, monkeypatch):
    import solution.experiment
    calls = []
    def fake(cfg, checkpoint, split, data, work):
        calls.append((checkpoint, split, data, work))
        return tmp_path/"pred.jsonl"
    monkeypatch.setattr(solution.experiment, "infer", fake)
    output = tmp_path/"result.json"
    execute({"operation": "infer", "config": {}, "checkpoint": "cp", "split": "test",
             "data_dir": "fixture-data", "work_dir": "fixture-work", "result_path": str(output)})
    assert calls == [("cp", "test", "fixture-data", "fixture-work")]
    assert read_json(output)["predictions"] == str(tmp_path/"pred.jsonl")
