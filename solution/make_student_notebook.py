"""Build a self-contained Colab notebook from reviewed source files."""
import base64
import io
import json
import zipfile
from pathlib import Path


def source_archive():
    paths = [p for directory in ("solution", "scripts", "tests", "student_tests", "configs")
             for p in Path(directory).rglob("*") if p.suffix in (".py", ".yaml")]
    paths += [Path(n) for n in ("README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md", "RUNBOOK.md",
                                "requirements-colab.txt", "requirements-student.txt", "pytest.ini", ".gitignore")]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.writestr(path.as_posix(), path.read_bytes())
    return buffer.getvalue()


def build():
    cells = []
    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.splitlines(True)})
    md("# Lab Day 10 — Nguyễn Thị Thùy Dương\n\n"
       "Qwen3-VL-4B: E0–E4 + bonus E5. Mã nguồn được đóng gói trong notebook; công cụ đề giữ nguyên.\n"
       "Chọn **T4 GPU**, chạy các cell lần lượt. Kết quả chỉ xuất hiện sau khi huấn luyện/suy luận thật.\n")
    md("## 0. Môi trường (Colab T4)\nGiải nén source và cài đúng versions của đề.\n")
    payload = base64.b64encode(source_archive()).decode()
    code("import base64, io, os, sys, zipfile, subprocess\nfrom pathlib import Path\n"
         "ROOT = Path('/content/lab_day10')\nROOT.mkdir(exist_ok=True)\n"
         f"SOURCE_B64 = {payload!r}\n"
         "with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:\n    z.extractall(ROOT)\n"
         "os.chdir(ROOT)\nsys.path.insert(0, str(ROOT))\n"
         "# Put this notebook in ROOT too, so the instructor's notebook-structure test can run.\n"
         "from solution.make_student_notebook import build\nbuild()\n"
         "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-student.txt'], check=True)\n"
         "import torch\nassert torch.cuda.is_available(), 'Runtime → Change runtime type → T4 GPU'\n"
         "print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30, 'GiB')\n")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n"
         "WORK = Path('/content/drive/MyDrive/Day10_ThuyDuong/checkpoints')\nWORK.mkdir(parents=True, exist_ok=True)\n"
         "ARTIFACTS = WORK.parent / 'report'\nARTIFACTS.mkdir(exist_ok=True)\n"
         "if not Path('report').exists():\n    Path('report').symlink_to(ARTIFACTS, target_is_directory=True)\n"
         "print('Checkpoint:', WORK, '\\nEvidence:', ARTIFACTS)\n"
         "subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests', 'student_tests'], check=True)\n")
    md("## 1. Dữ liệu: tải và chia split\nDataset công khai tải qua Kagglehub; split do công cụ gốc của đề tạo.\n")
    code("import kagglehub\nraw = kagglehub.dataset_download('lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes')\n"
         "subprocess.run([sys.executable, 'scripts/prepare_data.py', '--source', raw], check=True)\n"
         "from solution.experiment import validate_data\nprint(validate_data())\n")
    md("## 2. Nhãn: YOLO → JSON thang 0–1000\nKiểm tra nhãn một ảnh và vẽ box để đối chiếu.\n")
    code("from scripts.labcommon import read_jsonl\nfrom solution.core import objects_to_answer, prompt\n"
         "from solution.reporting import annotate\nfrom IPython.display import display\nfrom PIL import Image\n"
         "train = read_jsonl('data/splits/train.jsonl')\n"
         "for i, entry in enumerate(train[:6]):\n    answer = objects_to_answer(entry['objects'])\n"
         "    annotate(entry, answer, 'data', f'report/figures/ground_truth_{i}.png')\n"
         "print(prompt())\nprint(objects_to_answer(train[0]['objects']))\ndisplay(Image.open('report/figures/ground_truth_0.png'))\n")
    md("## 3. E0: zero-shot\nE0 dùng model gốc NF4; output thô được lưu nguyên văn. Pipeline ở bước 7 chạy và chấm E0 trước huấn luyện.\n")
    code("from solution.core import load_config\nfrom solution.modeling import load_base, architecture_counts\n"
         "from solution.experiment import infer, score_run\nprint(load_config('configs/e0.yaml'))\n")
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss\nChỉ answer/EOS có label; prefix và padding=-100. Hàm kiểm tra tiền tố chính xác trước khi train.\n")
    code("from solution.modeling import encode_example, collate, encoding_audit\n"
         "import inspect\nprint(inspect.getsource(encode_example))\nprint(inspect.getsource(collate))\n")
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3\nLLM 252 targets; E3 thêm 96 ViT targets. E2/E3 lưu riêng projector full-tune.\n")
    code("from solution.modeling import apply_setup\nprint(inspect.getsource(apply_setup))\n"
         "for i in (1,2,3):\n    print(load_config(f'configs/e{i}.yaml'))\n")
    md("## 6. Vòng huấn luyện\nAdamW, warmup 5%, cosine, clipping 1.0, fp16/autocast, fp32 master weights; checkpoint 50 bước.\n")
    code("from solution.training import train_sft\nprint(inspect.getsource(train_sft))\n"
         "# Smoke test có tên riêng, không được dùng làm evidence nộp bài.\n"
         "# subprocess.run([sys.executable, '-m', 'solution.experiment', 'train', '--config', 'configs/e1.yaml', '--steps', '2', '--work', str(WORK)], check=True)\n")
    md("## 7. Suy luận và chấm E0–E3; chạy E4 và E5\n"
       "Cell này chạy toàn bộ pipeline và tiếp tục checkpoint nếu bị ngắt. Có thể cần nhiều giờ trên T4.\n"
       "Chọn baseline bằng validation; không dùng test để chọn setup.\n")
    code("subprocess.run([sys.executable, '-u', '-m', 'solution.experiment', 'suite', '--work', str(WORK)], check=True)\n")
    md("## 8. E4 (ablation dữ liệu)\n"
       "10% và 100% của setup chọn bằng validation; giữ mọi biến khác. 100% tái sử dụng đúng baseline.\n")
    code("from solution.storage import read_json\nprint(read_json('report/e4_design.json'))\n"
         "print(read_json('report/evidence/e4_data_size_0.1.json'))\nprint(read_json('report/evidence/e4_data_size_1.0.json'))\n")
    md("## 9. Bonus E5 — GRPO\n"
       "Reward có test: hoàn hảo, JSON lỗi, sai lớp, thừa box, rỗng, giant box. "
       "4 generations, 128 tokens, 60 bước, use_vllm=False; so với SFT bằng số thật.\n")
    code("from solution.core import iou_reward\nprint(inspect.getsource(iou_reward))\n"
         "subprocess.run([sys.executable, '-m', 'pytest', '-q', 'student_tests/test_core.py'], check=True)\n"
         "print(read_json('report/evidence/e5.json'))\n")
    md("## 10. Báo cáo, kiểm tra và tải kết quả\n"
       "Sinh báo cáo từ evidence; checker báo FAIL nếu chưa đủ kết quả hoặc chưa đạt ngưỡng rubric.\n")
    code("subprocess.run([sys.executable, '-m', 'solution.reporting', '--work', str(WORK)], check=True)\n"
         "import shutil\nshutil.copy2('REPORT.md', WORK.parent/'REPORT.md')\n"
         "subprocess.run([sys.executable, '-m', 'solution.check_quality'], check=True)\n"
         "# Git commits kiểm tra trên repo local khi đưa kết quả về; runtime giải nén không có .git.\n"
         "subprocess.run([sys.executable, 'scripts/check_submission.py'], check=True)\n"
         "from google.colab import files\n"
         "with zipfile.ZipFile('/content/day10_results.zip','w',zipfile.ZIP_DEFLATED) as z:\n"
         "    z.write('REPORT.md')\n    for p in Path('report').rglob('*'):\n        if p.is_file(): z.write(p, p.as_posix())\n"
         "files.download('/content/day10_results.zip')\n"
         "display(Image.open('report/figures/learning_and_f1.png'))\n")
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
    Path("lab_day10.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    Path("work").mkdir(exist_ok=True)
    Path("work/lab_source.zip").write_bytes(source_archive())


if __name__ == "__main__":
    build()
