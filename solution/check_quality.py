"""Rubric checks beyond the instructor's structural submission checker."""
import argparse
from pathlib import Path

from solution.storage import read_json


def checks(root="."):
    root = Path(root)
    results = []
    def add(ok, message):
        results.append((bool(ok), message))
    evidence = {}
    for name in ("e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0", "e5"):
        path = root/"report/evidence"/f"{name}.json"
        if not path.exists():
            add(False, f"Missing actual evidence: {name}")
            continue
        d = read_json(path)
        evidence[name] = d
        add(d.get("split") == "test" and d.get("score_version") == 1 and
            d.get("missing_predictions") == 0, f"Complete official test evidence: {name}")
    if "e0" in evidence and "e1" in evidence:
        add(evidence["e1"]["parse_valid_rate"] >= .95, "E1 parse_valid_rate >= 0.95")
        add(evidence["e1"]["micro"]["f1"]-evidence["e0"]["micro"]["f1"] >= .10,
            "E1 F1 exceeds E0 by at least 0.10")
    add(len({d.get("dataset_version") for d in evidence.values()}) == 1 and bool(evidence),
        "All experiments use one dataset_version")
    low_path = root/"report/configs/e4_data_size_0.1.json"
    high_path = root/"report/configs/e4_data_size_1.0.json"
    if low_path.exists() and high_path.exists():
        low, high = read_json(low_path), read_json(high_path)
        add(low["train_fraction"] == .1 and high["train_fraction"] == 1, "Both E4 data-size levels")
        for c in (low, high):
            for k in ("name", "description", "ablation", "train_fraction"):
                c.pop(k, None)
        add(low == high, "E4 changes only dataset fraction")
    else:
        add(False, "Both resolved E4 configs present")
    for name in ("e1", "e2", "e3", "e4_data_size_0.1", "e5"):
        path = root/"report/runs"/f"{name}.json"
        expected = 60 if name == "e5" else 300
        add(path.exists() and read_json(path).get("step") == expected, f"Actual {expected} steps: {name}")
    add(len(list((root/"report/figures").glob("prediction_*.png"))) >= 6, "At least six annotated predictions")
    add((root/"report/logs/e5.jsonl").exists(), "Actual bonus training log")
    add((root/"report/paired_session_bootstrap.json").exists(), "Paired confidence intervals by session")
    report = root/"REPORT.md"
    add(report.exists() and "<điền" not in report.read_text(encoding="utf-8"), "Report uses measured results")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    results = checks(args.root)
    for ok, message in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {message}")
    raise SystemExit(0 if all(ok for ok, _ in results) else 1)


if __name__ == "__main__":
    main()
