"""Run the lab, preserving raw outputs and using the official scorer unchanged."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

from scripts.labcommon import read_jsonl
from scripts.score import main as official_score
from solution.core import ablations, fingerprint, load_config, objects_to_answer
from solution.modeling import load_base, open_image, user_message
from solution.storage import append_jsonl, load_projector, read_json, write_json
from solution.training import train_grpo, train_sft


def validate_data(data_dir="data"):
    root = Path(data_dir)
    seen_images, seen_sessions, counts = set(), {}, {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(root / f"splits/{split}.jsonl")
        boxes = {c: 0 for c in ("pothole", "crack", "manhole")}
        for row in rows:
            if row["image_id"] in seen_images:
                raise ValueError("Duplicate image across splits")
            seen_images.add(row["image_id"])
            previous = seen_sessions.setdefault(row["session"], split)
            if previous != split:
                raise ValueError("Session leakage across splits")
            image = open_image(row, root)
            if image.size != (row["width"], row["height"]):
                raise ValueError("Image dimensions differ from metadata")
            objects_to_answer(row["objects"])
            for obj in row["objects"]:
                boxes[obj["label"]] += 1
        counts[split] = {"images": len(rows), "boxes": boxes}
    meta = read_json(root / "splits/meta.json")
    if counts != meta["counts"]:
        raise ValueError("Actual dataset differs from meta counts")
    write_json("report/data_audit.json", {"dataset_version": meta["dataset_version"],
                                         "counts": counts, "n_sessions": len(seen_sessions),
                                         "session_leakage": False})
    return counts


def read_predictions(path):
    """Repair only an interrupted trailing JSONL line, never alter model text."""
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = {}
    offset = 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines)-1:
                raise ValueError("Corrupt prediction file before final line") from None
            with path.open("r+b") as f:
                f.truncate(offset)
            break
        if row["image_id"] in rows:
            raise ValueError("Duplicate prediction id")
        rows[row["image_id"]] = row["raw"]
        offset += len(line)
    # An intact final object with no newline must be separated from the next appended row.
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n")
    return rows


def infer(cfg, checkpoint=None, split="test", data_dir="data", work_dir="work", batch_size=1):
    import torch
    from peft import PeftModel
    entries = read_jsonl(Path(data_dir)/f"splits/{split}.jsonl")
    out = Path(work_dir) / f"{cfg['name'].lower()}_{split}.jsonl"
    manifest_path = out.with_suffix(".manifest.json")
    manifest = {"config": cfg, "dataset_version": read_json(Path(data_dir)/"splits/meta.json")["dataset_version"],
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_run": read_json(Path(checkpoint)/"run.json") if checkpoint else None,
                "split": split, "image_ids": [e["image_id"] for e in entries]}
    identity = fingerprint(manifest)
    if manifest_path.exists():
        if read_json(manifest_path)["identity"] != identity:
            raise ValueError("Predictions belong to a different model/dataset/config")
    elif out.exists():
        raise ValueError("Prediction file has no provenance manifest")
    else:
        write_json(manifest_path, dict(manifest, identity=identity))
    done = read_predictions(out)
    if set(done) - set(manifest["image_ids"]):
        raise ValueError("Unexpected image id in prediction file")
    pending = [e for e in entries if e["image_id"] not in done]
    if not pending:
        return out
    model, processor = load_base(cfg)
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        load_projector(model, checkpoint, cfg.get("train_projector", False))
    model.eval()
    model.config.use_cache = True
    start = time.perf_counter()
    try:
        for offset in range(0, len(pending), batch_size):
            rows = pending[offset:offset+batch_size]
            messages = [[user_message(open_image(e, data_dir), e, cfg)] for e in rows]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
            tail = output[:, inputs["input_ids"].shape[-1]:]
            raw = processor.batch_decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for e, text in zip(rows, raw):
                append_jsonl(out, {"image_id": e["image_id"], "raw": text})
            if offset % 20 == 0:
                print(f"{cfg['name']} {split}: {len(done)+offset+len(rows)}/{len(entries)}", flush=True)
            del inputs, output, tail
        write_json(out.with_suffix(".timing.json"), {"new_images": len(pending),
                                                    "inference_minutes": (time.perf_counter()-start)/60})
        return out
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def score_run(cfg, predictions, split, state=None, data_dir="data", evidence_dir="report/evidence"):
    output = Path(evidence_dir)/f"{cfg['name'].lower()}.json" if split == "test" else (
        Path("report/validation")/f"{cfg['name'].lower()}.json")
    args = ["--pred", str(predictions), "--splits-dir", str(Path(data_dir)/"splits"),
            "--split", split, "--coord", cfg["coord"], "--name", cfg["name"], "--out", str(output)]
    for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
        if state:
            args.extend(["--meta", f"{key}={state[key]}"])
    if cfg.get("baseline"):
        args.extend(["--meta", f"baseline={cfg['baseline']}"])
    official_score(args)
    return read_json(output)


def run_suite(data_dir="data", work_dir="work", bonus=True, steps=None):
    validate_data(data_dir)
    configs = [load_config(f"configs/e{i}.yaml") for i in range(4)]
    # Effective batch stays 8; use the documented T4 fallback for every SFT arm.
    for cfg in configs:
        cfg["max_new_tokens"] = 512
        if "lora" in cfg:
            cfg.update(batch_size=1, grad_accum=8)
            if steps is not None:
                cfg["max_steps"] = steps
    if steps is not None:
        raise ValueError("Use individual 'train' for smoke runs; the submission suite requires 300 steps")
    runs = {}
    for cfg in configs:
        write_json(Path("report/configs")/f"{cfg['name'].lower()}.json", cfg)
        checkpoint, state = train_sft(cfg, data_dir, work_dir) if "lora" in cfg else (None, None)
        validation = score_run(cfg, infer(cfg, checkpoint, "val", data_dir, work_dir), "val", state, data_dir)
        score_run(cfg, infer(cfg, checkpoint, "test", data_dir, work_dir), "test", state, data_dir)
        runs[cfg["name"]] = {"cfg": cfg, "checkpoint": str(checkpoint) if checkpoint else None,
                             "state": state, "validation": validation}
    best = max((runs[f"E{i}"] for i in (1, 2, 3)),
               key=lambda r: (r["validation"]["micro"]["f1"], -r["state"]["trainable_params"]))
    write_json("report/selection.json", {"criterion": "highest full-validation F1, tie: fewer trainable params",
                                         "selected": best["cfg"]["name"],
                                         "validation_f1": {n: r["validation"]["micro"]["f1"] for n, r in runs.items()}})
    low, high = ablations(best["cfg"], "data_size")
    for cfg in (low, high):
        write_json(Path("report/configs")/f"{cfg['name'].lower()}.json", cfg)
    checkpoint, state = train_sft(low, data_dir, work_dir)
    score_run(low, infer(low, checkpoint, "val", data_dir, work_dir), "val", state, data_dir)
    score_run(low, infer(low, checkpoint, "test", data_dir, work_dir), "test", state, data_dir)
    # 100% is exactly the baseline; reuse raw predictions and disclose this in metadata.
    baseline_predictions = Path(work_dir)/f"{best['cfg']['name'].lower()}_test.jsonl"
    score_run(high, baseline_predictions, "test", best["state"], data_dir)
    write_json("report/e4_design.json", {"variable": "data_size", "values": [0.1, 1.0],
                                         "baseline": best["cfg"]["name"], "reused_100pct_baseline": True,
                                         "constant_optimizer_steps": 300,
                                         "note": "Smaller dataset is repeated more often; this measures data volume at fixed compute."})
    if bonus:
        checkpoint, state = train_grpo(best["cfg"], best["checkpoint"], load_config("configs/e5.yaml"), data_dir, work_dir)
        cfg = state["config"]
        write_json("report/configs/e5.json", cfg)
        score_run(cfg, infer(cfg, checkpoint, "val", data_dir, work_dir), "val", state, data_dir)
        score_run(cfg, infer(cfg, checkpoint, "test", data_dir, work_dir), "test", state, data_dir)
    write_json("report/suite_complete.json", {"bonus": bonus, "work_dir": str(work_dir),
                                              "selected": best["cfg"]["name"], "status": "completed"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    suite = sub.add_parser("suite")
    suite.add_argument("--data", default="data")
    suite.add_argument("--work", default="work")
    suite.add_argument("--no-bonus", action="store_true")
    single = sub.add_parser("train")
    single.add_argument("--config", required=True)
    single.add_argument("--data", default="data")
    single.add_argument("--work", default="work")
    single.add_argument("--steps", type=int)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "suite":
        run_suite(args.data, args.work, bonus=not args.no_bonus)
    else:
        cfg = load_config(args.config)
        cfg.update(batch_size=1, grad_accum=8)
        if args.steps:
            cfg.update(max_steps=args.steps, name=cfg["name"]+f"_smoke{args.steps}")
        train_sft(cfg, args.data, args.work)


if __name__ == "__main__":
    main()
