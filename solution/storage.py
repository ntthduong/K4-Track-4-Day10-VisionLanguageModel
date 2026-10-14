"""Atomic JSON and checkpoint persistence."""
import json
import math
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def append_training_log(path, row):
    """Preserve undefined telemetry as null; fail explicitly on invalid training metrics."""
    non_finite = {}

    def clean(value, field):
        if isinstance(value, float) and not math.isfinite(value):
            non_finite[field] = str(value)
            return None
        if isinstance(value, dict):
            return {k: clean(v, f'{field}.{k}' if field else k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v, f'{field}[{i}]') for i, v in enumerate(value)]
        return value

    sanitized = clean(row, '')
    if non_finite:
        sanitized['non_finite_metrics'] = non_finite
    append_jsonl(path, sanitized)
    critical = {'loss', 'grad_norm', 'reward', 'rewards/reward_function/mean'} & non_finite.keys()
    if critical:
        raise FloatingPointError(f'Non-finite training metrics: {sorted(critical)}; diagnostic row saved to {path}')


def save_projector(model, directory):
    from safetensors.torch import save_file
    from solution.modeling import is_projector
    state = {n: p.detach().float().cpu().contiguous() for n, p in model.named_parameters() if is_projector(n)}
    if state:
        save_file(state, str(Path(directory) / "projector.safetensors"))


def load_projector(model, directory, required):
    import torch
    from safetensors.torch import load_file
    from solution.modeling import is_projector
    path = Path(directory) / "projector.safetensors"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing trained projector: {path}")
        return
    state = load_file(str(path))
    params = dict(model.named_parameters())
    expected = {n for n in params if is_projector(n)}
    if set(state) != expected:
        raise RuntimeError("Projector checkpoint keys differ from the loaded model")
    with torch.no_grad():
        for n, tensor in state.items():
            if params[n].shape != tensor.shape:
                raise RuntimeError(f"Projector shape mismatch: {n}")
            params[n].copy_(tensor.to(device=params[n].device, dtype=params[n].dtype))
