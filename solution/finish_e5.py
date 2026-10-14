"""Run only bonus E5 from the saved Kaggle SFT checkpoint and export real results."""
import argparse
import shutil
import zipfile
from pathlib import Path

from solution.core import load_config
from solution.experiment import isolated, score_run, validate_data
from solution.reporting import build_report
from solution.storage import read_json, write_json


def finish(previous, work='checkpoints', data='data'):
    previous, work = Path(previous), Path(work)
    validate_data(data)
    selection = read_json('report/selection.json')
    name = selection['selected'].lower()
    cfg = read_json(f'report/configs/{name}.json')
    expected = read_json(f'report/runs/{name}.json')
    candidates = []
    for run in (previous/'checkpoints'/name).glob('*/run.json'):
        state = read_json(run)
        if state.get('step') == 300 and state.get('run_id') == expected['run_id']:
            cp = run.parent
            if (cp/'adapter_model.safetensors').exists() and (cp/'projector.safetensors').exists():
                candidates.append(cp)
    if not candidates:
        raise FileNotFoundError(f'No complete {name.upper()} step-300 adapter/projector in {previous}; refuse to train E0-E4 again')
    checkpoint = sorted(candidates)[-1]
    print(f'E5_BASE_CHECKPOINT={checkpoint}', flush=True)
    work.mkdir(parents=True, exist_ok=True)
    for pred in (previous/'checkpoints').glob('*_test.jsonl'):
        shutil.copy2(pred, work/pred.name)
    for run in Path('report/runs').glob('*.json'):
        target = work/run.stem/'run.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run, target)
    for log in (previous/'report/logs').glob('*.jsonl'):
        target = work/log.stem/'train_log.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, target)
    # Resume an existing E5 checkpoint if the failed output actually contains one.
    old_e5 = previous/'checkpoints/e5'
    if old_e5.exists() and not (work/'e5').exists():
        shutil.copytree(old_e5, work/'e5')
    trained = isolated('grpo', cfg, data, work, gpu_device=0, checkpoint=str(checkpoint),
                       e5_config=load_config('configs/e5.yaml'))
    state, final = trained['state'], trained['checkpoint']
    write_json('report/runs/e5.json', state)
    write_json('report/configs/e5.json', state['config'])
    Path('report/logs').mkdir(parents=True, exist_ok=True)
    shutil.copy2(work/'e5/train_log.jsonl', 'report/logs/e5.jsonl')
    for split in ('val', 'test'):
        result = isolated('infer', state['config'], data, work, gpu_device=0,
                          checkpoint=final, split=split)
        score_run(state['config'], result['predictions'], split, state, data)
    build_report(data_dir=data, work_dir=work)
    with zipfile.ZipFile('day10_e5_results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write('REPORT.md')
        for path in Path('report').rglob('*'):
            if path.is_file():
                archive.write(path)
        for path in work.glob('*_test.jsonl'):
            archive.write(path, f'predictions/{path.name}')
    print('E5_TRAIN_EVALUATION_AND_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--work', default='checkpoints')
    parser.add_argument('--data', default='data')
    args = parser.parse_args()
    finish(args.previous, args.work, args.data)
