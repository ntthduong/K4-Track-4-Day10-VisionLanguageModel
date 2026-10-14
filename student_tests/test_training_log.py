import json

import pytest

from solution.storage import append_training_log


def test_undefined_telemetry_is_preserved_without_breaking_training(tmp_path):
    path = tmp_path/'train.jsonl'
    append_training_log(path, {'step': 18, 'loss': .2, 'completions/mean_terminated_length': float('nan')})
    row = json.loads(path.read_text())
    assert row['loss'] == .2
    assert row['completions/mean_terminated_length'] is None
    assert row['non_finite_metrics'] == {'completions/mean_terminated_length': 'nan'}


@pytest.mark.parametrize('field', ['loss', 'grad_norm', 'reward', 'rewards/reward_function/mean'])
def test_invalid_training_metric_is_logged_then_fails(tmp_path, field):
    path = tmp_path/'train.jsonl'
    with pytest.raises(FloatingPointError, match='Non-finite training metrics'):
        append_training_log(path, {'step': 18, field: float('nan')})
    assert json.loads(path.read_text())['non_finite_metrics'][field] == 'nan'
