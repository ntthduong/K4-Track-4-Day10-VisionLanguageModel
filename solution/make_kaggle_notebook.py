"""Build the full lab notebook for a Kaggle Save & Run All session."""
import json
from pathlib import Path

from solution.make_student_notebook import build


def build_kaggle():
    build()
    notebook = json.loads(Path("lab_day10.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[0]["source"] = [
        "# Lab Day 10 — Nguyễn Thị Thùy Dương (Kaggle)\n",
        "\nChọn **GPU T4 x2**, bật **Internet**, rồi **Save Version → Save & Run All**.\n",
        "E0–E3 chạy trên hai GPU riêng; E4 và bonus E5 dùng checkpoint chọn bằng validation.\n",
        "Mỗi model dùng một GPU; không thay batch hiệu dụng hay số bước huấn luyện.\n",
        "Checkpoint, report và day10_results.zip nằm trong /kaggle/working/lab_day10/.\n",
        "Kết quả được lưu trong output của phiên Save & Run All; không cần Drive.\n",
    ]
    cells[1]["source"] = ["## 0. Môi trường (Kaggle T4 x2)\nGiải nén source và cài đúng versions của đề.\n"]
    bootstrap = "".join(cells[2]["source"]).replace("/content/lab_day10", "/kaggle/working/lab_day10")
    bootstrap = bootstrap.replace("ROOT.mkdir(exist_ok=True)\n",
        "ROOT.mkdir(exist_ok=True)\n"
        "# Keep the 8.9 GB base model cache outside saved notebook outputs.\n"
        "os.environ['HF_HOME'] = '/kaggle/temp/day10_hf_cache'\n")
    bootstrap = bootstrap.replace("Runtime → Change runtime type → T4 GPU", "Settings → Accelerator → GPU T4 x2")
    bootstrap += "assert torch.cuda.device_count() >= 2, 'Choose GPU T4 x2 before Save & Run All'\n"
    cells[2]["source"] = bootstrap.splitlines(True)
    cells[3]["source"] = (
        "WORK = ROOT/'checkpoints'\nWORK.mkdir(exist_ok=True)\n"
        "Path('report').mkdir(exist_ok=True)\n"
        "print('Checkpoint:', WORK, '\\nSaved output:', ROOT)\n"
        "subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests', 'student_tests'], check=True)\n"
    ).splitlines(True)
    cells[5]["source"] = (
        "import kagglehub\n"
        "raw = Path('/kaggle/input/road-damage-dataset-potholes-cracks-and-manholes')\n"
        "if not raw.exists():\n"
        "    raw = Path(kagglehub.dataset_download('lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes'))\n"
        "subprocess.run([sys.executable, 'scripts/prepare_data.py', '--source', str(raw)], check=True)\n"
        "from solution.experiment import validate_data\nprint(validate_data())\n"
    ).splitlines(True)
    cells[16]["source"] += [
        "\nKaggle: E0/E2 trên GPU 0; E1/E3 trên GPU 1. E4/E5 chạy tuần tự sau khi chọn baseline.\n",
        "CPU dùng chung: thời gian wall-clock không dùng để so sánh tốc độ giữa setup.\n",
    ]
    cells[17]["source"] = (
        "subprocess.run([sys.executable, '-u', '-m', 'solution.finish', '--work', str(WORK), "
        "'--gpus', '0', '1'], check=True)\n"
    ).splitlines(True)
    cells[22]["source"] = [
        "## 10. Báo cáo, kiểm tra và tải kết quả\n",
        "Pipeline đã sinh report và checker từ evidence thật.\n",
        "Sau khi phiên Save & Run All hoàn tất, tải lab_day10/day10_results.zip trong Output.\n",
    ]
    cells[23]["source"] = (
        "print(Path('REPORT.md').read_text(encoding='utf-8'))\n"
        "print(read_json('report/quality_checks.json'))\n"
        "print(Path('report/submission_check.txt').read_text())\n"
        "print('DOWNLOAD:', ROOT/'day10_results.zip')\n"
        "display(Image.open('report/figures/learning_and_f1.png'))\n"
    ).splitlines(True)
    notebook["metadata"].pop("colab", None)
    Path("lab_day10_kaggle.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build_kaggle()
