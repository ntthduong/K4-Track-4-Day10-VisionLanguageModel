"""Verify recovered E3 predictions and create six measured prediction/GT figures."""
import argparse
import hashlib
import math
import shutil
import statistics
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _match, _prf, evaluate, parse_output
from solution.core import gt_boxes
from solution.reporting import annotate
from solution.storage import read_json, write_json


def complete(pred_path, data_dir='data'):
    data = Path(data_dir)
    meta = read_json(data/'splits/meta.json')
    entries = read_jsonl(data/'splits/test.jsonl')
    rows = read_jsonl(pred_path)
    preds = {r['image_id']: r['raw'] for r in rows}
    assert len(rows) == len(preds) == len(entries) == 311
    assert set(preds) == {e['image_id'] for e in entries}
    evidence = read_json('report/evidence/e3.json')
    assert meta['dataset_version'] == evidence['dataset_version']
    measured = evaluate(entries, preds, coord='norm1000')
    for key in ('n_images', 'missing_predictions', 'parse_valid_rate', 'micro', 'per_class', 'f1_ci95', 'fp_per_image'):
        assert measured[key] == evidence[key], f'Recovered E3 differs from evidence: {key}'
    assert math.isclose(measured['mean_iou'], evidence['mean_iou'], abs_tol=1e-12)
    ranking = []
    for e in entries:
        per, _ = _match(gt_boxes(e['objects']), parse_output(preds[e['image_id']]) or [])
        tp, fp, fn = [sum(v[k] for v in per.values()) for k in range(3)]
        ranking.append((_prf(tp, fp, fn)['f1'], e['image_id'], e))
    ranking.sort(key=lambda r: (r[0], r[1]))
    chosen = ranking[:3] + ranking[-3:]
    samples = []
    for i, (f1, _, entry) in enumerate(chosen, 1):
        destination = f'report/figures/prediction_{i}.png'
        stats = annotate(entry, preds[entry['image_id']], data, destination)
        samples.append(dict(stats, f1=f1, figure=destination, raw=preds[entry['image_id']],
                            session=entry['session'], selection='lowest image F1' if i <= 3 else 'highest image F1'))
    destination = Path('report/predictions/e3_test.jsonl')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if Path(pred_path).resolve() != destination.resolve():
        shutil.copyfile(pred_path, destination)
    wrong = evaluate(entries, preds, coord='pixel')
    e0_example = read_json('report/e0_preview_example.json')
    e0_entry = next(e for e in entries if e['image_id'] == e0_example['image_id'])
    e0_stats = evaluate([e0_entry], {e0_entry['image_id']: e0_example['raw']})
    result = {
        'dataset_version': meta['dataset_version'],
        'n_images': len(entries), 'n_test_sessions': len({e['session'] for e in entries}),
        'source_predictions_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'official_score_matches_e3': True,
        'correct_f1': measured['micro']['f1'],
        'same_outputs_scored_as_pixel_f1': wrong['micro']['f1'],
        'e0_preview_micro': e0_stats['micro'],
        'median_gt_box_area_percent': {c: statistics.median(o['yolo'][2]*o['yolo'][3]*100 for e in entries for o in e['objects'] if o['label'] == c) for c in CLASSES},
        'samples': samples,
        'selection_policy': 'Three lowest and three highest per-image F1, ties by image_id; selected across all 311 test images.',
    }
    write_json('report/e3_image_analysis.json', result)
    print(f'Verified E3 F1={measured["micro"]["f1"]:.6f}; six annotated figures saved.')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--data', default='data')
    args = ap.parse_args()
    complete(args.pred, args.data)
