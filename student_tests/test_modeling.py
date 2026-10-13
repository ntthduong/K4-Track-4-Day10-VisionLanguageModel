import torch

from solution.modeling import collate


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
