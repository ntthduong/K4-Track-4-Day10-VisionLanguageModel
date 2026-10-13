"""Run every experiment, then export the completed submission artifacts."""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from solution.check_quality import checks
from solution.experiment import run_suite
from solution.reporting import build_report
from solution.storage import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--author", default="Nguyễn Thị Thùy Dương")
    ap.add_argument("--gpus", nargs="+", type=int)
    args = ap.parse_args()
    run_suite(work_dir=args.work, bonus=True, gpu_devices=args.gpus)
    build_report(work_dir=args.work, author=args.author)
    destination = Path(args.work).resolve().parent
    if Path("REPORT.md").resolve() != (destination/"REPORT.md").resolve():
        shutil.copy2("REPORT.md", destination/"REPORT.md")
    quality = checks()
    write_json("report/quality_checks.json", [{"passed": ok, "check": message} for ok, message in quality])
    structural = subprocess.run([sys.executable, "scripts/check_submission.py"], text=True, capture_output=True)
    print(structural.stdout, structural.stderr, flush=True)
    Path("report/submission_check.txt").write_text(structural.stdout+structural.stderr, encoding="utf-8")
    with zipfile.ZipFile(destination/"day10_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write("REPORT.md")
        for p in Path("report").rglob("*"):
            if p.is_file():
                archive.write(p, p.as_posix())
    for ok, message in quality:
        print(f"[{'PASS' if ok else 'FAIL'}] {message}", flush=True)
    print(f"RESULTS_ARCHIVE={destination/'day10_results.zip'}", flush=True)
    if structural.returncode or not all(ok for ok, _ in quality):
        raise SystemExit("Real runs finished, but some rubric requirements need investigation; see quality_checks.json")
    print("ALL_EXPERIMENTS_AND_BONUS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
