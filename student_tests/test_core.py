import json

import pytest

from solution.core import (ablations, gt_boxes, iou_reward, load_config,
                           objects_to_answer, reward_function, subset_train, yolo_to_box)
from scripts.score import evaluate


GT = [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]
PERFECT = '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'


def test_coordinate_roundtrip_and_scale_trap():
    entry = {"image_id": "a", "width": 640, "height": 360, "objects": GT}
    for coord in ("norm1000", "pixel"):
        answer = objects_to_answer(GT, coord)
        result = evaluate([entry], {"a": answer}, coord=coord, n_boot=20)
        assert result["micro"]["f1"] == 1
    wrong = evaluate([entry], {"a": objects_to_answer(GT, "pixel")}, n_boot=20)
    assert wrong["micro"]["f1"] == 0


def test_stable_order_and_empty():
    objs = GT + [{"label": "crack", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    assert objects_to_answer(objs) == objects_to_answer(list(reversed(objs)))
    assert json.loads(objects_to_answer(objs))[0]["label"] == "crack"
    assert objects_to_answer([]) == "[]"


def test_zero_area_training_target_does_not_change_original_scoring_gt():
    zero = {"label": "crack", "yolo": [0.639062, 0.786111, 0.003125, 0.0]}
    objects = GT + [zero]
    assert objects_to_answer(objects) == PERFECT
    assert objects == GT + [zero]
    assert len(gt_boxes(objects)) == 2
    assert gt_boxes(objects)[1]["bbox"][1] == gt_boxes(objects)[1]["bbox"][3]
    with pytest.raises(ValueError):
        objects_to_answer([{"label": "alien", "yolo": [0.5, 0.5, 0, 0]}])


@pytest.mark.parametrize("yolo", [[0.5, 0.5, 0, 0.1], [float("nan"), 0.5, 0.1, 0.1],
                                  [1.1, 0.5, 0.1, 0.1], [0.5, 0.5, 0.00001, 0.1]])
def test_invalid_or_collapsed_box_is_loud(yolo):
    with pytest.raises(ValueError):
        yolo_to_box("pothole", yolo)


def test_reward_perfect_invalid_wrong_class_extra_and_empty():
    boxes = gt_boxes(GT)
    assert iou_reward(PERFECT, boxes) == pytest.approx(1)
    assert iou_reward("garbage", boxes) == 0
    assert iou_reward(PERFECT.replace("pothole", "crack"), boxes) == 0.2
    duplicate = json.dumps(json.loads(PERFECT)*2)
    assert iou_reward(duplicate, boxes) == pytest.approx(0.6)
    assert iou_reward("[]", []) == 1
    assert iou_reward("[]", boxes) == 0.2
    assert iou_reward(PERFECT, []) == 0.2


def test_giant_box_and_unknown_class_do_not_get_iou_credit():
    assert iou_reward('[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]', gt_boxes(GT)) == 0.2
    assert iou_reward(PERFECT.replace("pothole", "alien"), gt_boxes(GT)) == 0.2


def test_trl_conversational_reward_adapter():
    assert reward_function([[{"role": "assistant", "content": PERFECT}]], [gt_boxes(GT)], prompt=[]) == [1]


def test_ablation_changes_one_factor_and_subset_reproducible():
    base = load_config("configs/e2.yaml")
    low, high = ablations(base)
    assert low["train_fraction"] == 0.1 and high["train_fraction"] == 1
    for key in ("lora", "train_projector", "lr", "batch_size", "grad_accum", "max_steps", "coord"):
        assert low[key] == high[key] == base[key]
    entries = [{"image_id": str(i)} for i in range(100)]
    assert len(subset_train(entries, 0.1)) == 10
    assert subset_train(entries, 0.1) == subset_train(entries, 0.1)
    assert subset_train(entries, 1) == entries
