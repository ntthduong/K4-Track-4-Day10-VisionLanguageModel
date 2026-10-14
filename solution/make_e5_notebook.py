"""Package patched source and measured metadata for E5-only Kaggle continuation."""
import base64
import io
import json
import zipfile
from pathlib import Path

from solution.make_student_notebook import source_archive


def build():
    payload = io.BytesIO(source_archive())
    with zipfile.ZipFile(payload, 'a', zipfile.ZIP_DEFLATED) as archive:
        for p in Path('report').rglob('*.json'):
            archive.writestr(p.as_posix(), p.read_bytes())
    packed = base64.b64encode(payload.getvalue()).decode()
    setup = '''import base64, io, os, sys, zipfile, subprocess
from pathlib import Path
ROOT = Path('/kaggle/working/lab_day10_e5')
ROOT.mkdir(exist_ok=True)
os.environ['HF_HOME'] = '/kaggle/temp/day10_hf_cache'
'''
    setup += 'SOURCE_B64 = (\n' + ''.join(f'    {packed[i:i+100]!r}\n' for i in range(0,len(packed),100)) + ')\n'
    setup += '''with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as archive:
    archive.extractall(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-student.txt'], check=True)
from solution.colab_environment import prepare
prepare()
import torch
assert torch.cuda.is_available(), 'Enable GPU T4 and Internet'
print(torch.cuda.get_device_name(0))
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'student_tests/test_training_log.py', 'student_tests/test_core.py'], check=True)
'''
    run = '''import json
previous_roots = [p.parent.parent for p in Path('/kaggle/input').rglob('selection.json') if p.parent.name == 'report' and (p.parent.parent/'checkpoints/e3/checkpoint-300').exists()]
assert len(previous_roots) == 1, f'Attach Full Bonus notebook Version 2 output: {previous_roots}'
PREVIOUS = previous_roots[0]
print('PREVIOUS_OUTPUT=', PREVIOUS)
raw_candidates = [p.parent for p in Path('/kaggle/input').rglob('labels-YOLO') if p.is_dir() and (p.parent/'images').is_dir()]
if raw_candidates:
    data_command = ['--source', str(raw_candidates[0])]
else:
    data_command = ['--zip', 'https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes']
subprocess.run([sys.executable, 'scripts/prepare_data.py', *data_command], check=True)
assert json.loads(Path('data/splits/meta.json').read_text())['dataset_version'] == 'd10-21e4070301'
resume_roots = sorted({p.parent.parent.parent.parent for p in Path('/kaggle/input').rglob('trainer_state.json') if p.parent.name.startswith('checkpoint-') and p.parent.parent.name == 'e5'})
resume_options = ['--resume-output', str(resume_roots[-1])] if resume_roots else []
print('E5_RESUME_OUTPUT=', resume_roots)
subprocess.run([sys.executable, '-u', '-m', 'solution.finish_e5', '--previous', str(PREVIOUS), *resume_options], check=True)
'''
    def cell(kind, text):
        result = {'cell_type':kind, 'metadata':{}, 'source':text.splitlines(True)}
        if kind == 'code': result.update(execution_count=None, outputs=[])
        return result
    notebook = {'nbformat':4, 'nbformat_minor':5, 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}, 'cells':[
        cell('markdown', '# E5 GRPO continuation — Nguyễn Thị Thùy Dương\n\nAttach Full Bonus Version 2 notebook output and original Road Damage Dataset. Enable Internet and GPU T4. Only E5 trains; E0–E4 checkpoints/results are reused. Undefined telemetry is recorded as null; invalid loss/gradient/reward fails explicitly. Checkpoint every five steps.'),
        cell('code',setup), cell('code',run),
        cell('code', "print(Path('REPORT.md').read_text())\nprint('DOWNLOAD:', ROOT/'day10_e5_results.zip')\nprint(Path('report/evidence/e5.json').read_text())\n") ]}
    path = Path('lab_day10_e5_kaggle.ipynb')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
