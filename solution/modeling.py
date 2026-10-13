"""Qwen3-VL encoding, exact LoRA targets, and loss-safe multimodal collation."""
import re
from pathlib import Path

from PIL import Image

from solution.core import objects_to_answer, prompt


LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"
UNQUANTIZED_MODULES = ["model.visual", "visual", "lm_head"]


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment needs a CUDA GPU. Select T4 in Colab Runtime settings.")


def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base(cfg):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    require_gpu()
    seed_everything(cfg["seed"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=UNQUANTIZED_MODULES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    processor.tokenizer.padding_side = "left"
    for name, p in model.named_parameters():
        if ("visual." in name or "lm_head." in name) and p.__class__.__name__ == "Params4bit":
            raise RuntimeError(f"ViT was quantized unexpectedly: {name}")
    return model, processor


def architecture_counts(model):
    """Count original logical tensor shapes, including packed 4-bit weights."""
    counts = {"vit_without_projector": 0, "projector": 0, "llm_and_head": 0}
    for name, p in model.named_parameters():
        shape = getattr(getattr(p, "quant_state", None), "shape", p.shape)
        import math
        n = math.prod(shape)
        key = "projector" if is_projector(name) else (
            "vit_without_projector" if "visual." in name else "llm_and_head")
        counts[key] += n
    counts["total"] = sum(counts.values())
    return counts


def configure_trainable(model, cfg):
    import torch
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name or (cfg.get("train_projector", False) and is_projector(name)))
        if p.requires_grad:
            # GradScaler cannot unscale fp16 gradients; projector and LoRA masters are fp32.
            p.data = p.data.to(torch.float32)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False


def apply_setup(model, cfg):
    from peft import LoraConfig, get_peft_model
    pattern = LLM_TARGET
    expected = model.config.text_config.num_hidden_layers * 7
    if cfg["lora"]["targets"] == "llm+vit":
        pattern = f"(?:{LLM_TARGET}|{VIT_TARGET})"
        expected += model.config.vision_config.depth * 4
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    if len(matched) != expected:
        raise RuntimeError(f"LoRA matched {len(matched)} modules; expected {expected}. Inspect model tree.")
    for p in model.parameters():
        p.requires_grad_(False)
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"],
                                           lora_dropout=lc["dropout"], target_modules=pattern,
                                           task_type="CAUSAL_LM", bias="none"))
    configure_trainable(model, cfg)
    return model, matched


def user_message(image, entry, cfg):
    return {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt(cfg["coord"], entry["width"], entry["height"])}]}


def open_image(entry, data_dir):
    with Image.open(Path(data_dir) / entry["image_path"]) as im:
        return im.convert("RGB")


def encode_example(entry, processor, data_dir, cfg):
    import torch
    image = open_image(entry, data_dir)
    user = user_message(image, entry, cfg)
    answer = objects_to_answer(entry["objects"], cfg["coord"], entry["width"], entry["height"])
    options = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prefix = processor.apply_chat_template([user], add_generation_prompt=True, **options)
    full = processor.apply_chat_template([user, {"role": "assistant", "content": answer}],
                                         add_generation_prompt=False, **options)
    n = prefix["input_ids"].shape[-1]
    if not torch.equal(full["input_ids"][0, :n], prefix["input_ids"][0]):
        raise RuntimeError("Chat prompt is not a prefix: do not train with an incorrect loss mask")
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :n] = -100
    if not (full["labels"] != -100).any():
        raise RuntimeError("No assistant tokens to supervise")
    return dict(full)


def collate(examples, pad_token_id):
    import torch
    if not examples:
        raise ValueError("empty batch")
    length = max(e["input_ids"].shape[-1] for e in examples)
    result = {}
    for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0),
                     ("mm_token_type_ids", 0), ("labels", -100)):
        if key not in examples[0]:
            continue
        rows = []
        for e in examples:
            # Right padding during SFT; generation uses processor left padding.
            rows.append(torch.nn.functional.pad(e[key], (0, length-e[key].shape[-1]), value=pad))
        result[key] = torch.cat(rows, dim=0)
    for key in ("pixel_values", "image_grid_thw"):
        if key in examples[0]:
            result[key] = torch.cat([e[key] for e in examples], dim=0)
    unknown = set(examples[0]) - set(result)
    if unknown:
        raise RuntimeError(f"Unrecognized processor outputs: {unknown}; inspect collation before training")
    return result


def encoding_audit(entry, processor, data_dir, cfg):
    enc = encode_example(entry, processor, data_dir, cfg)
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_image = int((enc["input_ids"] == image_id).sum())
    if entry["width"] == 640 and entry["height"] == 360 and n_image != 220:
        raise RuntimeError(f"Expected 220 visual tokens for 640x360, found {n_image}")
    selected = enc["input_ids"][enc["labels"] != -100]
    return {"image_id": entry["image_id"], "vision_tokens": n_image,
            "assistant_tokens": int(selected.numel()),
            "supervised_text": processor.tokenizer.decode(selected, skip_special_tokens=False)}
