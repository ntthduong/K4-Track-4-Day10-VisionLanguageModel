"""SFT with real resource logs, deterministic resume, and explicit projector saves."""
import gc
import importlib.metadata
import math
import random
import shutil
import time
from pathlib import Path

from scripts.labcommon import read_jsonl
from solution.core import fingerprint, subset_train
from solution.modeling import (apply_setup, architecture_counts, collate, configure_trainable,
                               encode_example, encoding_audit, load_base)
from solution.storage import append_jsonl, append_training_log, load_projector, read_json, save_projector, write_json


def environment():
    import torch
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "versions": {p: importlib.metadata.version(p) for p in
                         ("transformers", "peft", "trl", "accelerate", "bitsandbytes")}}


def batch_indices(n, size, cursor, seed):
    """Each complete pass uses a reproducible permutation; keep a cursor for resume."""
    result = []
    while len(result) < size:
        epoch, offset = divmod(cursor, n)
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        take = min(size-len(result), n-offset)
        result.extend(order[offset:offset+take])
        cursor += take
    return result, cursor


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def save_checkpoint(model, processor, opt, scheduler, scaler, run_dir, state):
    import numpy as np
    import torch
    checkpoint = run_dir / f"checkpoint-{state['step']}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    if state["config"].get("train_projector"):
        save_projector(model, checkpoint)
    torch.save({"optimizer": cpu_tree(opt.state_dict()), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, checkpoint / "training_state.pt")
    write_json(checkpoint / "run.json", state)
    # Publish pointer only after all files are complete. Previous pointer survives an interrupted save.
    write_json(run_dir / "latest.json", {"directory": checkpoint.name})
    for old in run_dir.glob("checkpoint-*"):
        if old != checkpoint and old.is_dir() and old.resolve().parent == run_dir.resolve():
            shutil.rmtree(old)
    return checkpoint


def train_sft(cfg, data_dir="data", work_dir="work", resume=True):
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import get_cosine_schedule_with_warmup
    run_dir = Path(work_dir) / cfg["name"].lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_meta = read_json(Path(data_dir) / "splits/meta.json")
    run_id = fingerprint({"config": cfg, "dataset_version": dataset_meta["dataset_version"]})
    previous = None
    if (run_dir / "latest.json").exists():
        if not resume:
            raise FileExistsError("Run exists; use resume or a new experiment name")
        checkpoint = run_dir / read_json(run_dir / "latest.json")["directory"]
        previous = read_json(checkpoint / "run.json")
        if previous["run_id"] != run_id:
            raise ValueError("Existing run used a different config/dataset; give this experiment a new name")
        if previous["step"] >= cfg["max_steps"]:
            return checkpoint, previous
    entries = subset_train(read_jsonl(Path(data_dir) / "splits/train.jsonl"), cfg["train_fraction"], cfg["seed"])
    model, processor = load_base(cfg)
    architecture = architecture_counts(model)
    if previous:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
        load_projector(model, checkpoint, cfg.get("train_projector", False))
        configure_trainable(model, cfg)
        matched = previous["matched_modules"]
    else:
        model, matched = apply_setup(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    audit = encoding_audit(entries[0], processor, data_dir, cfg)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(opt, math.ceil(cfg["max_steps"]*0.05), cfg["max_steps"])
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    state = {"name": cfg["name"], "config": cfg, "run_id": run_id,
             "dataset_version": dataset_meta["dataset_version"], "step": 0, "cursor": 0,
             "trainable_params": sum(p.numel() for p in params), "architecture": architecture,
             "matched_modules": matched, "encoding_audit": audit, "train_minutes": 0.0,
             "peak_vram_gb": 0.0, "n_train_images": len(entries), "environment": environment()}
    state["model_revision"] = getattr(model.config, "_commit_hash", None)
    if previous:
        state = previous
        # Only load locally produced training state. It includes Python/NumPy RNG tuples.
        saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    torch.cuda.reset_peak_memory_stats()
    start, elapsed_before = time.perf_counter(), state["train_minutes"]
    model.train()
    opt.zero_grad(set_to_none=True)
    overflow_retries = 0
    print(f"{cfg['name']}: {state['trainable_params']:,} trainable params, {len(matched)} LoRA modules", flush=True)
    try:
        while state["step"] < cfg["max_steps"]:
            cursor_before = state["cursor"]
            rng_before = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            losses = []
            for _ in range(cfg["grad_accum"]):
                indices, state["cursor"] = batch_indices(len(entries), cfg["batch_size"], state["cursor"], cfg["seed"])
                batch = collate([encode_example(entries[i], processor, data_dir, cfg) for i in indices],
                                processor.tokenizer.pad_token_id)
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    raw_loss = model(**batch).loss
                    loss = raw_loss / cfg["grad_accum"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; inspect labels and reduce LR before a new run")
                losses.append(float(raw_loss.detach()))
                scaler.scale(loss).backward()
                del batch, loss, raw_loss
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm):
                # Dynamic loss scaling may overflow despite finite forward loss.
                # Skip the optimizer update, lower scale, and replay the same examples.
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                state["cursor"] = cursor_before
                torch.set_rng_state(rng_before[0])
                torch.cuda.set_rng_state_all(rng_before[1])
                overflow_retries += 1
                print(f"Gradient overflow: retry {overflow_retries}, loss scale={scaler.get_scale()}", flush=True)
                if overflow_retries >= 8:
                    raise FloatingPointError("Repeated gradient overflow; previous complete checkpoint remains available")
                continue
            overflow_retries = 0
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["train_minutes"] = elapsed_before + (time.perf_counter()-start)/60
            state["peak_vram_gb"] = max(state["peak_vram_gb"], torch.cuda.max_memory_allocated()/2**30)
            row = {"step": state["step"], "loss": sum(losses)/len(losses),
                   "lr": scheduler.get_last_lr()[0], "grad_norm": float(norm),
                   "train_minutes": state["train_minutes"], "peak_vram_gb": state["peak_vram_gb"]}
            append_jsonl(run_dir / "train_log.jsonl", row)
            if state["step"] % 10 == 0 or state["step"] == 1:
                print(row, flush=True)
            if state["step"] % 50 == 0 or state["step"] == cfg["max_steps"]:
                checkpoint = save_checkpoint(model, processor, opt, scheduler, scaler, run_dir, state)
        write_json(run_dir / "run.json", state)
        return checkpoint, state
    finally:
        del model, processor, opt, scheduler, scaler, params
        gc.collect()
        torch.cuda.empty_cache()


def train_grpo(base_cfg, checkpoint, e5_cfg, data_dir="data", work_dir="work"):
    """Continue the selected SFT adapter; projector is frozen for the bonus run."""
    import torch
    from peft import PeftModel
    from torch.utils.data import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from solution.core import gt_boxes, prompt, reward_function, training_objects
    from solution.modeling import open_image
    cfg = dict(base_cfg, name="E5", base_setup=base_cfg["name"], grpo=e5_cfg["grpo"])
    run_dir = Path(work_dir) / "e5"
    if (run_dir / "run.json").exists():
        old = read_json(run_dir / "run.json")
        if old["config"] != cfg:
            raise ValueError("E5 already exists with different config")
        if old["step"] >= cfg["grpo"]["max_steps"]:
            return run_dir / "final", old
    entries = read_jsonl(Path(data_dir) / "splits/train.jsonl")
    model, processor = load_base(base_cfg)
    architecture = architecture_counts(model)
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    load_projector(model, checkpoint, base_cfg.get("train_projector", False))
    configure_trainable(model, dict(base_cfg, train_projector=False))
    class RoadDataset(Dataset):
        def __len__(self):
            return len(entries)

        def __getitem__(self, index):
            e = entries[index]
            return {"prompt": [{"role": "user", "content": prompt()}],
                    "image": open_image(e, data_dir), "ground_truth": gt_boxes(training_objects(e["objects"])),
                    "image_id": e["image_id"]}

    class SaveExtra(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            target = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            # Save frozen but previously SFT-trained projector, too.
            if base_cfg.get("train_projector"):
                save_projector(kwargs["model"], target)
            write_json(target / "resource.json", {
                "train_minutes": prior_minutes+(time.perf_counter()-start)/60,
                "peak_vram_gb": max(prior_peak, torch.cuda.max_memory_allocated()/2**30)})

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                append_training_log(run_dir / "train_log.jsonl", dict(logs, step=state.global_step))

    g = cfg["grpo"]
    args = GRPOConfig(output_dir=str(run_dir), max_steps=g["max_steps"],
                      learning_rate=g["learning_rate"], beta=g["beta"],
                      num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
                      per_device_train_batch_size=1, gradient_accumulation_steps=4,
                      steps_per_generation=4, generation_batch_size=None,
                      fp16=True, bf16=False, gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={"use_reentrant": False}, use_vllm=False,
                      remove_unused_columns=False, report_to="none", logging_steps=1,
                      save_steps=5, save_total_limit=2, seed=cfg["seed"],
                      dataloader_num_workers=0, optim="adamw_torch", max_grad_norm=1.0)
    trainer = GRPOTrainer(model=model, processing_class=processor, args=args,
                          train_dataset=RoadDataset(), reward_funcs=reward_function,
                          callbacks=[SaveExtra()])
    # TRL 1.9.2 unconditionally casts QLoRA trainables to bf16 in __init__.
    # T4 needs fp16 autocast and fp32 master gradients; restore after construction.
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    checkpoint_dirs = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume_path = str(checkpoint_dirs[-1]) if checkpoint_dirs else None
    resource_path = Path(resume_path)/"resource.json" if resume_path else None
    prior = read_json(resource_path) if resource_path and resource_path.exists() else {}
    prior_minutes, prior_peak = prior.get("train_minutes", 0.0), prior.get("peak_vram_gb", 0.0)
    if resume_path and base_cfg.get("train_projector"):
        load_projector(model, resume_path, True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=resume_path)
        final = run_dir / "final"
        trainer.save_model(str(final))
        processor.save_pretrained(final)
        if base_cfg.get("train_projector"):
            save_projector(model, final)
        state = {"name": "E5", "config": cfg, "step": trainer.state.global_step,
                 "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                 "peak_vram_gb": max(prior_peak, torch.cuda.max_memory_allocated()/2**30),
                 "train_minutes": prior_minutes+(time.perf_counter()-start)/60,
                 "architecture": architecture, "metrics": result.metrics,
                 "dataset_version": read_json(Path(data_dir)/"splits/meta.json")["dataset_version"],
                 "environment": environment(), "resume_from_checkpoint": resume_path}
        write_json(final / "run.json", state)
        write_json(run_dir / "run.json", state)
        return final, state
    finally:
        del trainer, model, processor
        gc.collect()
        torch.cuda.empty_cache()
