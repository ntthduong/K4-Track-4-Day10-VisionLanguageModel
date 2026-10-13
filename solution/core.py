"""Coordinate conversion, reward, and experimental design (no GPU imports)."""
import copy
import hashlib
import json
import math
import random
from pathlib import Path

from scripts.labcommon import CLASSES
from scripts.score import iou_xyxy, parse_output


def prompt(coord="norm1000", width=640, height=360):
    if coord not in ("norm1000", "pixel"):
        raise ValueError(coord)
    scale = ("integer coordinates from 0 to 1000, normalized independently by image width and height"
             if coord == "norm1000" else f"integer pixel coordinates; image width={width}, height={height}")
    return ("Detect every visible pothole, crack, and manhole in this road image. "
            "Use only these labels: pothole, crack, manhole. "
            f"Return bounding boxes as [x1,y1,x2,y2] using {scale}. "
            'Return ONLY a JSON list: [{"label":"pothole","bbox_2d":[x1,y1,x2,y2]}]. '
            "Order objects top to bottom, then left to right. Return [] if none. "
            "Do not add explanations, markdown, or objects outside the image.")


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    if label not in CLASSES or coord not in ("norm1000", "pixel"):
        raise ValueError("unknown label or coordinate convention")
    if len(yolo) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in yolo):
        raise ValueError("YOLO values must be finite and in [0,1]")
    xc, yc, w, h = yolo
    if w <= 0 or h <= 0:
        raise ValueError("zero-area ground truth")
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    box = [round(max(0, min(1, v)) * s) for v, s in zip(
        (xc-w/2, yc-h/2, xc+w/2, yc+h/2), (sx, sy, sx, sy))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("box collapsed after rounding; inspect this annotation")
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda o: (o["bbox_2d"][1], o["bbox_2d"][0], o["label"], o["bbox_2d"]))
    return json.dumps(boxes, separators=(",", ":"))


def gt_boxes(objects):
    """Unrounded norm1000 GT, matching the official scorer."""
    result = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        result.append({"label": o["label"], "bbox": (
            (xc-w/2)*1000, (yc-h/2)*1000, (xc+w/2)*1000, (yc+h/2)*1000)})
    return result


def iou_reward(completion, ground_truth):
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not pred and not ground_truth:
        return 1.0
    pairs = sorted((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                   for i, g in enumerate(ground_truth) for j, p in enumerate(pred)
                   if g["label"] == p["label"])
    used_gt, used_pred, total = set(), set(), 0.0
    for overlap, i, j in reversed(pairs):
        if overlap < 0.5:
            break
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
            total += overlap
    return 0.2 + 0.8 * total / max(len(ground_truth), len(pred), 1)


def reward_function(completions, ground_truth, **kwargs):
    """TRL passes extra dataset columns to reward functions."""
    if len(completions) != len(ground_truth):
        raise ValueError("completion/GT length mismatch")
    def text(c):
        if isinstance(c, str):
            return c
        return "".join(m.get("content", "") for m in c)
    return [iou_reward(text(c), gt) for c, gt in zip(completions, ground_truth)]


def subset_train(entries, fraction, seed=42):
    if not 0 < fraction <= 1 or not entries:
        raise ValueError("empty dataset or invalid fraction")
    indices = list(range(len(entries)))
    random.Random(seed).shuffle(indices)
    selected = set(indices[:max(1, int(len(entries)*fraction))])
    return [e for i, e in enumerate(entries) if i in selected]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("coord", "norm1000")
    cfg.setdefault("max_new_tokens", 512)
    cfg.setdefault("train_fraction", 1.0)
    cfg.setdefault("seed", 42)
    return cfg


def ablations(base, variable="data_size"):
    values = {"rank": [4, 64], "data_size": [0.1, 1.0],
              "coord_scale": ["norm1000", "pixel"]}[variable]
    result = []
    for value in values:
        cfg = copy.deepcopy(base)
        cfg.update(name=f"E4_{variable}_{value}", baseline=base["name"],
                   ablation={"variable": variable, "value": value})
        if variable == "rank":
            cfg["lora"].update(r=value, alpha=2*value)
        elif variable == "data_size":
            cfg["train_fraction"] = value
        else:
            cfg["coord"] = value
        result.append(cfg)
    return result
