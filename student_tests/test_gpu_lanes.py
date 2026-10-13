import os
import threading
import time

import pytest

from solution import experiment
from solution.storage import read_json, write_json


def test_unequal_duration_jobs_never_share_a_gpu(monkeypatch):
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active, calls = set(), []

    def fake_setup(cfg, data, work, gpu):
        with lock:
            assert gpu not in active
            active.add(gpu)
            calls.append((cfg["name"], gpu))
        if cfg["name"] in ("E0", "E1"):
            barrier.wait(timeout=5)
        time.sleep(0.06 if cfg["name"] == "E0" else 0.005)
        with lock:
            active.remove(gpu)
        return {"fixture": cfg["name"]}

    monkeypatch.setattr(experiment, "run_setup", fake_setup)
    configs = [{"name": f"E{i}"} for i in range(4)]
    result = experiment.initial_setups(configs, "fixture-data", "fixture-work", [0, 1])
    assert set(result) == {"E0", "E1", "E2", "E3"}
    assert sorted(calls) == [("E0", 0), ("E1", 1), ("E2", 0), ("E3", 1)]
    assert not active


@pytest.mark.parametrize("devices", [[0, 0], [-1], ["0"], [True]])
def test_invalid_gpu_plan_cannot_launch_jobs(monkeypatch, devices):
    monkeypatch.setattr(experiment, "run_setup", lambda *a: pytest.fail("invalid GPU plan launched"))
    with pytest.raises(ValueError):
        experiment.initial_setups([{"name": "E0"}], "data", "work", devices)


def test_gpu_environment_is_per_child_not_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-value")
    seen = []

    def fake_run(command, **options):
        request = read_json(command[-1])
        seen.append(options["env"]["CUDA_VISIBLE_DEVICES"])
        assert request["gpu_device"] == 1
        write_json(request["result_path"], {"predictions": "fixture-predictions"})

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    result = experiment.isolated("infer", {"name": "E0"}, "data", tmp_path, gpu_device=1, split="test")
    assert result["predictions"] == "fixture-predictions"
    assert seen == ["1"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-value"
