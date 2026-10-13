import torch
import pytest
from PIL import Image

from solution.modeling import collate
from solution.modeling import encode_example


def test_collation_masks_padding_and_concatenates_visual_grids():
    a = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
         "mm_token_type_ids": torch.tensor([[0, 1, 0]]), "labels": torch.tensor([[-100, -100, 3]]),
         "pixel_values": torch.ones(4, 6), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    b = {"input_ids": torch.tensor([[1, 4]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
         "mm_token_type_ids": torch.tensor([[0, 1]]), "labels": torch.tensor([[-100, 4]]),
         "pixel_values": torch.zeros(2, 6), "image_grid_thw": torch.tensor([[1, 1, 2]])}
    batch = collate([a, b], 0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [1, 4, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 3], [-100, 4, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["pixel_values"].shape == (6, 6)
    assert batch["image_grid_thw"].shape == (2, 3)


def test_loss_mask_includes_answer_and_end_token_only(tmp_path):
    Image.new("RGB", (640, 360)).save(tmp_path/"sample.jpg")
    entry = {"image_path": "sample.jpg", "width": 640, "height": 360, "objects": []}
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            ids = [10, 11, 12] if len(messages) == 1 else [10, 11, 12, 20, 21]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
    cfg = {"coord": "norm1000"}
    encoded = encode_example(entry, Processor(), tmp_path, cfg)
    assert encoded["labels"].tolist() == [[-100, -100, -100, 20, 21]]
    class BrokenProcessor(Processor):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if len(messages) == 2:
                result["input_ids"][0, 0] = 99
            return result
    with pytest.raises(RuntimeError, match="not a prefix"):
        encode_example(entry, BrokenProcessor(), tmp_path, cfg)
