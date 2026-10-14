"""Verify the saved Kaggle E5 archive, import evidence and regenerate the report."""
import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

from scripts.labcommon import read_jsonl
from scripts.score import evaluate
from solution.experiment import read_predictions
from solution.reporting import build_report
from solution.storage import read_json, write_json


def import_results(archive_path, work='work/e5_results_recovery'):
    archive_path, destination = Path(archive_path), Path(work).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None, 'Corrupt archive'
        assert all((destination / p).resolve().is_relative_to(destination)
                   for p in archive.namelist()), 'Unsafe archive path'
        archive.extractall(destination)
    entries = read_jsonl('data/splits/test.jsonl')
    version = read_json('data/splits/meta.json')['dataset_version']
    comparisons = {}
    for name in ('e0', 'e1', 'e2', 'e3', 'e4_data_size_0.1', 'e5'):
        evidence = read_json(destination / f'report/evidence/{name}.json')
        predictions = read_predictions(destination / f'predictions/{name}_test.jsonl')
        assert set(predictions) == {e['image_id'] for e in entries}
        scored = evaluate(entries, predictions, coord='norm1000', n_boot=1000)
        for key, value in scored.items():
            assert value == evidence[key], (name, key, value, evidence[key])
        assert evidence['dataset_version'] == version
        if name != 'e5':
            assert evidence == read_json(f'report/evidence/{name}.json'), name
        comparisons[name] = {'n_predictions': len(predictions), 'all_score_fields_match': True}
    for p in (destination / 'report').rglob('*.json'):
        relative = p.relative_to(destination)
        local = Path(relative)
        if local.exists() and relative.parts[1] in ('runs', 'configs', 'evidence'):
            assert read_json(p) == read_json(local), str(relative)
    state = read_json(destination / 'report/runs/e5.json')
    assert state['step'] == 60 and state['config']['base_setup'] == 'E3'
    assert state['dataset_version'] == version
    rows = read_jsonl(destination / 'report/logs/e5.jsonl')
    latest = {r['step']: r for r in rows if 'loss' in r}
    assert set(latest) == set(range(1, 61))
    for row in latest.values():
        for metric in ('loss', 'grad_norm', 'reward'):
            assert isinstance(row[metric], (int, float)) and math.isfinite(row[metric])
        assert not row.get('optimizer_step_skipped', False)
    for step in range(6, 61):
        assert latest[step]['loss_scale'] == 128.0
    # Import only new measured results; inherited verification files are stale.
    for relative in ('report/evidence/e5.json', 'report/validation/e5.json',
                     'report/runs/e5.json', 'report/configs/e5.json', 'report/logs/e5.jsonl'):
        target = Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination / relative, target)
    Path('report/predictions').mkdir(parents=True, exist_ok=True)
    for prediction in (destination / 'predictions').glob('*.jsonl'):
        shutil.copy2(prediction, Path('report/predictions') / prediction.name)
    report_work = destination / 'report_work'
    report_work.mkdir(exist_ok=True)
    for prediction in (destination / 'predictions').glob('*.jsonl'):
        shutil.copy2(prediction, report_work / prediction.name)
    for run in Path('report/runs').glob('*.json'):
        target = report_work / run.stem / 'run.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run, target)
    shutil.copy2('report/logs/e5.jsonl', report_work / 'e5/train_log.jsonl')
    build_report('data', report_work)
    # Confirm derived diagnostics against the original Kaggle result, too.
    for relative in ('report/diagnostics.json', 'report/paired_session_bootstrap.json'):
        assert read_json(relative) == read_json(destination / relative), relative
    write_json('report/e5_provenance.json', {
        'notebook_url': 'https://www.kaggle.com/code/ntthduong/day10-qwen3vl-thuyduong-e5-stable/notebook',
        'kaggle_session_id': '358221873', 'version': 1, 'runtime_seconds': 2355,
        'archive': archive_path.name,
        'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'official_scorer_reproduction': comparisons,
        'diagnostics_and_paired_ci_match_kaggle': True,
        'completed_steps': 60, 'training_log_rows': len(rows),
        'latest_finite_loss_grad_reward_steps': len(latest),
        'resumed_steps_with_scale_128': 55,
        'historical_non_finite_rows': [r for r in rows if r.get('non_finite_metrics')],
        'validation_limit': 'Validation raw predictions are not in the ZIP; validation evidence is preserved as downloaded.',
        'loss_plot_limit': 'Only the E5 per-step log was included in the ZIP; E0-E4 final observed losses remain in observed_final_steps.json.'})
    print('All six raw test outputs reproduce official scores; E5 completed 60 finite steps.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    args = parser.parse_args()
    import_results(args.archive)
