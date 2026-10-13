"""Generate tables, annotated images and an evidence-based Vietnamese report."""
import argparse
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _match, _prf, evaluate, parse_output
from solution.core import gt_boxes, iou_reward
from solution.experiment import read_predictions
from solution.storage import read_json, write_json


def paired_delta(entries, a, b, coord_a="norm1000", coord_b="norm1000", n_boot=2000):
    """Paired session bootstrap: neighboring frames are not independent."""
    import random
    from scripts.score import _to_norm1000
    grouped = {}
    for e in entries:
        totals = []
        for preds, coord in ((a, coord_a), (b, coord_b)):
            pred = parse_output(preds.get(e["image_id"])) or []
            per, _ = _match(gt_boxes(e["objects"]), _to_norm1000(pred, coord, e["width"], e["height"]))
            totals.append([sum(v[k] for v in per.values()) for k in range(3)])
        group = grouped.setdefault(e["session"], [[0, 0, 0], [0, 0, 0]])
        for j in (0, 1):
            for k in range(3):
                group[j][k] += totals[j][k]
    groups, rng, samples = list(grouped.values()), random.Random(42), []
    for _ in range(n_boot):
        totals = [[0, 0, 0], [0, 0, 0]]
        for _ in groups:
            g = rng.choice(groups)
            for j in (0, 1):
                for k in range(3):
                    totals[j][k] += g[j][k]
        samples.append(_prf(*totals[1])["f1"]-_prf(*totals[0])["f1"])
    samples.sort()
    return {"n_sessions": len(groups), "delta_ci95": [samples[int(.025*n_boot)], samples[int(.975*n_boot)-1]],
            "method": "paired bootstrap by recording session; delta = B - A", "n_boot": n_boot}


def annotate(entry, raw, data_dir, destination, coord="norm1000"):
    from solution.modeling import open_image
    from scripts.score import _to_norm1000
    image = open_image(entry, data_dir).resize((1280, 720))
    canvas = Image.new("RGB", (1280, 800), "white")
    canvas.paste(image, (0, 80))
    draw = ImageDraw.Draw(canvas)
    gt = gt_boxes(entry["objects"])
    parsed = parse_output(raw)
    pred = _to_norm1000(parsed or [], coord, entry["width"], entry["height"])
    per, _ = _match(gt, pred)
    tp, fp, fn = [sum(v[k] for v in per.values()) for k in range(3)]
    draw.text((12, 12), f"{entry['image_id']} | TP={tp} FP={fp} FN={fn} | valid JSON={parsed is not None}", fill="black")
    draw.text((12, 35), "GREEN = ground truth; RED = prediction; raw output is preserved in JSONL", fill="black")
    for boxes, color, prefix in ((gt, "lime", "GT"), (pred, "red", "PRED")):
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            xy = [max(0, min(1279, x1*1.28)), max(80, min(799, 80+y1*.72)),
                  max(0, min(1279, x2*1.28)), max(80, min(799, 80+y2*.72))]
            draw.rectangle(xy, outline=color, width=3)
            draw.text((xy[0], max(80, xy[1]-12)), f"{prefix}:{b['label']}", fill=color, stroke_width=1, stroke_fill="black")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return {"image_id": entry["image_id"], "tp": tp, "fp": fp, "fn": fn, "parse_valid": parsed is not None}


def build_report(data_dir="data", work_dir="work", author="Nguyễn Thị Thùy Dương"):
    evidence = {p.stem: read_json(p) for p in Path("report/evidence").glob("*.json")}
    needed = {"e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0", "e5"}
    if not needed <= set(evidence):
        raise RuntimeError(f"Cannot write a completed report before real runs finish: missing {sorted(needed-set(evidence))}")
    meta = read_json(Path(data_dir)/"splits/meta.json")
    selected = read_json("report/selection.json")["selected"].lower()
    entries = read_jsonl(Path(data_dir)/"splits/test.jsonl")
    raw = {name: read_predictions(Path(work_dir)/f"{name}_test.jsonl")
           for name in ("e0", "e1", "e2", "e3", "e4_data_size_0.1", "e5")}
    raw["e4_data_size_1.0"] = raw[selected]
    cfg = read_json(f"report/configs/{selected}.json")
    diagnoses = {}
    for name in ("e0", selected, "e5"):
        pred = raw[name]
        invalid = sum(parse_output(v) is None for v in pred.values())
        empty = sum(parse_output(v) == [] for v in pred.values())
        unknown = sum(p["label"] not in CLASSES for v in pred.values() for p in (parse_output(v) or []))
        giant = sum((p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) >= 500000
                    for v in pred.values() for p in (parse_output(v) or []))
        avg_reward = statistics.mean(iou_reward(pred[e["image_id"]], gt_boxes(e["objects"])) for e in entries)
        diagnoses[name] = {"invalid_json_images": invalid, "empty_output_images": empty,
                           "unknown_label_boxes": unknown, "boxes_at_least_half_image": giant,
                           "mean_test_iou_reward": avg_reward}
    wrong_coord = evaluate(entries, raw[selected], coord="pixel", n_boot=1000)
    diagnoses["wrong_coordinate_scoring"] = {"correct_f1": evidence[selected]["micro"]["f1"],
                                             "same_raw_outputs_scored_as_pixel_f1": wrong_coord["micro"]["f1"]}
    paired = {}
    for a, b in (("e0", "e1"), ("e1", "e2"), ("e2", "e3"),
                 ("e4_data_size_0.1", "e4_data_size_1.0"), (selected, "e5")):
        paired[f"{a}_to_{b}"] = paired_delta(entries, raw[a], raw[b])
    write_json("report/diagnostics.json", diagnoses)
    write_json("report/paired_session_bootstrap.json", paired)
    ranking = []
    for e in entries:
        per, _ = _match(gt_boxes(e["objects"]), parse_output(raw[selected][e["image_id"]]) or [])
        tp, fp, fn = [sum(v[k] for v in per.values()) for k in range(3)]
        ranking.append((_prf(tp, fp, fn)["f1"], e["image_id"], e))
    ranking.sort(key=lambda x: (x[0], x[1]))
    samples = ranking[:3]+ranking[-3:]
    captions = []
    for i, (f1, _, e) in enumerate(samples):
        dest = Path("report/figures")/f"prediction_{i+1}.png"
        stats = annotate(e, raw[selected][e["image_id"]], data_dir, dest)
        captions.append(f"![{stats['image_id']}]({dest.as_posix()})\n\n"
                        f"Ảnh {i+1}: TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, "
                        f"F1 riêng ảnh={f1:.3f}. Box xanh là nhãn thật; đỏ là dự đoán.")
    # Resource/loss curves are drawn only from observed logs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("e1", "e2", "e3", "e4_data_size_0.1", "e5"):
        path = Path(work_dir)/name/"train_log.jsonl"
        if path.exists():
            rows = [r for r in read_jsonl(path) if "loss" in r]
            if rows:
                # Resumed jobs may repeat unsaved steps; latest recorded value wins.
                rows = list({r["step"]: r for r in rows}.values())
                rows.sort(key=lambda r: r["step"])
                axes[0].plot([r["step"] for r in rows], [r["loss"] for r in rows], label=name)
    axes[0].set(xlabel="Optimizer step", ylabel="Observed training loss")
    axes[0].legend()
    names = list(evidence)
    axes[1].errorbar(range(len(names)), [evidence[n]["micro"]["f1"] for n in names],
                     yerr=[[max(0, evidence[n]["micro"]["f1"]-evidence[n]["f1_ci95"][0]) for n in names],
                           [max(0, evidence[n]["f1_ci95"][1]-evidence[n]["micro"]["f1"]) for n in names]], fmt="o")
    axes[1].set_xticks(range(len(names)), names, rotation=45, ha="right")
    axes[1].set(ylabel="Test F1 / official image-bootstrap CI95", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig("report/figures/learning_and_f1.png", dpi=160)
    plt.close(fig)
    table = ["| Mã | Tham số học | VRAM GB | Phút | F1 | CI95 | FP/ảnh | pothole | crack | manhole |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for n in sorted(evidence):
        d, m = evidence[n], evidence[n].get("meta", {})
        ci = d["f1_ci95"]
        values = [d["name"], str(m.get("trainable_params", 0)), f"{m.get('peak_vram_gb',0):.2f}",
                  f"{m.get('train_minutes',0):.2f}", f"{d['micro']['f1']:.4f}", f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                  f"{d['fp_per_image']:.3f}"]+[f"{d['per_class'][c]['f1']:.4f}" for c in CLASSES]
        table.append("| "+" | ".join(values)+" |")
    arch = read_json(Path(work_dir)/"e1/run.json")["architecture"]
    split_lines = [f"- {s}: {c['images']} ảnh; box {c['boxes']}" for s, c in meta["counts"].items()]
    if Path("report/data_audit.json").exists():
        excluded = read_json("report/data_audit.json").get("zero_area_targets_excluded_from_training", [])
        if excluded:
            split_lines.append("- Nhãn có diện tích 0 chỉ được bỏ khỏi target SFT/GRPO; "
                               "split, số box gốc và ground truth chấm điểm giữ nguyên: "
                               + json.dumps(excluded, ensure_ascii=False))
    weakest = min(CLASSES, key=lambda c: evidence[selected]["per_class"][c]["f1"])
    size_stats = {c: statistics.median(o["yolo"][2]*o["yolo"][3]*100 for e in entries
                                     for o in e["objects"] if o["label"] == c) for c in CLASSES}
    audit = read_json(Path(work_dir)/"e1/run.json")["encoding_audit"]
    timing_note = (read_json("report/execution_plan.json").get("timing_note", "")
                   if Path("report/execution_plan.json").exists() else "")
    text = f"""Họ tên: {author}
`dataset_version`: {meta['dataset_version']}

## Bảng tổng hợp kết quả thật

{chr(10).join(table)}

{timing_note}

E0 không huấn luyện nên cột tài nguyên huấn luyện bằng 0. E4 100% tái sử dụng đúng run {selected.upper()}.
Tất cả SFT dùng batch 1 × tích luỹ 8 = batch hiệu dụng 8, 300 bước; suy luận greedy batch 16, tối đa 512 token cho mọi setup.
Chọn {selected.upper()} bằng F1 trên toàn bộ validation, tiêu chí và số liệu ở `report/selection.json`.

## 10 câu hỏi

### Câu 1 — Kiến trúc và phần được cập nhật
Đếm hình dạng tensor gốc: ViT không gồm projector {arch['vit_without_projector']:,}; projector {arch['projector']:,};
LLM và head {arch['llm_and_head']:,}; tổng {arch['total']:,} tham số.
E1 cập nhật LoRA LLM; E2 thêm toàn bộ merger và DeepStack mergers; E3 thêm LoRA attention/MLP ViT.
Số tham số học thực tế nằm trong bảng. ViT và lm_head giữ fp16; LLM dùng NF4 với double quantization.

### Câu 2 — E0 và lỗi quan sát được
`parse_valid_rate` = {evidence['e0']['parse_valid_rate']:.4f}. Thống kê thực tế: {diagnoses['e0']}.
Lỗi box/lớp được đo bằng FP={evidence['e0']['micro']['fp']} và FN={evidence['e0']['micro']['fn']}.
FP gồm box không đạt IoU hoặc sai lớp; FN là GT không có dự đoán cùng lớp đạt IoU 0.5.
Đây là mô tả quan sát, không gán lỗi sai thang chỉ dựa vào JSON hợp lệ.

### Câu 3 — Thang tọa độ và kiểm chứng
Nhãn là xyxy theo 0–1000, chuẩn hoá riêng theo chiều rộng/cao nên độc lập độ phân giải.
Chấm cùng output của {selected.upper()} đúng thang: F1={evidence[selected]['micro']['f1']:.4f};
cố ý chấm là pixel: F1={wrong_coord['micro']['f1']:.4f}. IoU thay đổi vì box bị co giãn sai.
Unit test còn kiểm chứng box YOLO [0.5,0.5,0.2,0.2] → [400,400,600,600] → pixel [256,144,384,216].

### Câu 4 — Mask loss
Chỉ assistant answer và token kết thúc được giám sát; prompt, token ảnh và padding có label=-100.
Kiểm tra prompt là tiền tố chính xác của chuỗi đầy đủ trước khi mask.
Audit một mẫu: {audit['vision_tokens']} token ảnh, {audit['assistant_tokens']} token assistant.
Tính loss trên prompt sẽ dành gradient cho phần người dùng thay vì tác vụ sinh nhãn; padding không mask gây bias token pad.
Chuỗi token được giám sát thực tế: `{audit['supervised_text']}`.

### Câu 5 — Projector và precision
Trọng số packed 4-bit không phải tham số float có thể tối ưu đầy đủ bằng AdamW. Vì vậy bỏ `visual` khỏi quantization.
Projector và LoRA có master weights float32 để GradScaler unscale gradient hợp lệ; forward vẫn chạy dưới autocast fp16.
Float32 không phải quy luật cho mọi phần cứng: ở đây đó là lựa chọn ổn định cho vòng AdamW/GradScaler trên T4.
Lưu riêng `projector.safetensors`, kiểm tra keys/shapes và nạp lại cùng adapter trước suy luận E2/E3/E5.

### Câu 6 — Chia theo phiên quay
Frame gần nhau có nội dung tương tự. Chia ngẫu nhiên tạo tương quan train/test, có thể làm F1 cao giả.
Audit không có image/session nằm ở hai split; dataset_version khớp mọi evidence.

{chr(10).join(split_lines)}

### Câu 7 — So sánh E1/E2/E3
Bảng trên ghi F1, CI95, tài nguyên thực tế. Setup được chọn bằng validation là {selected.upper()}.
So sánh độ tin cậy bằng paired bootstrap theo phiên quay (delta=B−A):
{json.dumps({k:v for k,v in paired.items() if k in ('e1_to_e2','e2_to_e3')}, ensure_ascii=False)}.
Nếu CI delta chứa 0, dữ liệu chưa chứng minh cải thiện; nếu toàn bộ CI >0, cải thiện có hỗ trợ từ các phiên test đã quan sát.
Đánh giá chi phí bằng VRAM/thời gian trong bảng; mở projector thêm gradient và optimizer state, E3 thêm backward qua ViT.

### Câu 8 — Độ tin cậy trên tập test nhỏ
Test có {len(entries)} ảnh nhưng chỉ {len(set(e['session'] for e in entries))} phiên quay.
CI chính thức bootstrap theo ảnh; frame cùng phiên tương quan nên CI đó có thể quá lạc quan.
`report/paired_session_bootstrap.json` bổ sung bootstrap theo phiên; số phiên ít cũng giới hạn khả năng khái quát.
Hai CI riêng lẻ chồng nhau không tự chứng minh không có khác biệt; dùng CI của chênh lệch ghép cặp.

### Câu 9 — Lỗi theo lớp
Ở {selected.upper()}, lớp có F1 thấp nhất là {weakest}: {evidence[selected]['per_class'][weakest]['f1']:.4f}.
Diện tích box trung vị trên test (% ảnh): {size_stats}. Số mẫu từng lớp có ở câu 6.
Box nhỏ làm IoU nhạy với sai lệch vài pixel; lượng mẫu và hình dạng nứt có thể góp phần nhưng không đủ chứng minh nguyên nhân.
Sáu hình dưới chọn 3 F1 ảnh thấp nhất và 3 cao nhất, không chỉ chọn ví dụ đẹp.

### Câu 10 — E4
Đổi duy nhất lượng dữ liệu train: 10% và 100% của {selected.upper()}; chọn subset cố định seed=42.
100% dùng lại baseline, các siêu tham số, prompt, thang tọa độ và số bước giữ nguyên.
F1 10%={evidence['e4_data_size_0.1']['micro']['f1']:.4f}; 100%={evidence['e4_data_size_1.0']['micro']['f1']:.4f}.
CI delta: {paired['e4_data_size_0.1_to_e4_data_size_1.0']['delta_ci95']}.
Do giữ 300 bước, 10% được lặp nhiều lần hơn; kết luận đo lượng dữ liệu với ngân sách bước cố định, không phải số epoch cố định.

## Bonus E5 — GRPO và reward hacking
Khởi tạo từ adapter {selected.upper()}, projector SFT giữ cố định; 4 generations, 128 completion tokens, 60 bước, không vLLM.
Reward: JSON lỗi=0; GT/pred cùng rỗng=1; còn lại 0.2+0.8×tổng IoU ghép cùng lớp đạt 0.5/max(nGT,nPred,1).
Test reward bao gồm hoàn hảo, JSON lỗi, sai lớp, thừa box, rỗng, nhãn lạ, và box khổng lồ.
Trước GRPO: {diagnoses[selected]}. Sau GRPO: {diagnoses['e5']}.
F1 SFT={evidence[selected]['micro']['f1']:.4f}, E5={evidence['e5']['micro']['f1']:.4f};
CI delta={paired[f'{selected}_to_e5']['delta_ci95']}.
Reward tăng nhưng F1 không tăng, hoặc tỷ lệ output rỗng/box khổng lồ tăng, là tín hiệu cần điều tra; không kết luận hacking chỉ từ một con số.
128 token có thể cắt output ảnh nhiều box; xem parse_valid_rate và raw outputs khi giải thích.

## Output E0 thực tế (5 mẫu)

```json
{json.dumps(list(raw['e0'].items())[:5], ensure_ascii=False, indent=2)}
```

## Hình minh hoạ

{chr(10).join(captions)}

![Loss và F1](report/figures/learning_and_f1.png)

## Nhìn lại
Kết quả là một seed trên dữ liệu hữu hạn; chưa đo độ biến thiên giữa nhiều seed hoặc nhiều thành phố.
Chia theo phiên giảm rò rỉ frame nhưng không đảm bảo các phiên không quay cùng tuyến đường.
Scorer hỗ trợ JSON bọc markdown; vì vậy parse_valid_rate không đồng nghĩa tuân thủ tuyệt đối yêu cầu chỉ JSON.
Những nguyên nhân lỗi hình ảnh cần kiểm tra bằng sáu ảnh và vấn đáp, không suy ra từ F1 đơn lẻ.

## Nguồn API đã đối chiếu
- [Qwen3-VL](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [GRPO TRL 1.9.2](https://huggingface.co/docs/trl/v1.9.2/en/grpo_trainer)
"""
    Path("REPORT.md").write_text(text, encoding="utf-8")
    return Path("REPORT.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--work", default="work")
    ap.add_argument("--author", default="Nguyễn Thị Thùy Dương")
    args = ap.parse_args()
    build_report(args.data, args.work, args.author)


if __name__ == "__main__":
    main()
