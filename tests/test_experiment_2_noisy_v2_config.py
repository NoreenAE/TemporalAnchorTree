"""Configuration plumbing only: never fit, predict, or generate data."""
from copy import deepcopy
import inspect
import json

import pandas as pd
import pytest

from experiments import experiment_2_noisy_v2 as v2
from experiments import experiment_2_fish_introduction_dynamic_anchors as shared
from settree.temporal_tree import TemporalTree


def test_configuration_preserves_every_constructor_parameter():
    assert v2.MODEL_CONFIG == shared.SHARED_MODEL_CONFIG
    assert v2.MODEL_CONFIG is not shared.SHARED_MODEL_CONFIG
    assert v2.MODEL_RUNS == shared.MODEL_RUNS
    assert list(v2.MODEL_CONFIG['feature_semantics']) == list(
        shared.SHARED_MODEL_CONFIG['feature_semantics'])
    parameters = set(inspect.signature(TemporalTree).parameters)
    assert set(v2.MODEL_CONFIG) | {'use_dynamic_anchors'} == parameters
    for enabled in (True, False):
        model = shared.make_model(enabled, v2.MODEL_CONFIG)
        assert model.get_params() == shared.make_model(enabled).get_params()
        model.anchor_offset_list.append(99)
        assert v2.MODEL_CONFIG['anchor_offset_list'] == [1, 2, 3, 4]


@pytest.mark.parametrize('name,value', [
    ('max_depth', 4), ('min_samples_split', 80), ('min_samples_leaf', 40),
    ('min_impurity_decrease', .01), ('candidate_sampling_rate', .7),
    ('random_state', 9), ('split_sampling_seed', 17),
    ('learn_missing_direction', True), ('verbose', 2),
    ('anchor_offset_list', [1, 3]),
    ('anchor_offset_map', {'Birth Date Anchor': [2, None]}),
])
def test_changes_reach_actual_model_before_fit(monkeypatch, tmp_path, name, value):
    config = deepcopy(v2.MODEL_CONFIG)
    config[name] = value
    data = pd.DataFrame({'patient_id': ['saved_patient']})
    patients = pd.DataFrame({'patient_id': ['saved_patient'],
                             'partition': ['train'], 'FA': [1]})

    class StopBeforeTraining(Exception):
        pass

    def stop(self, x, y):
        assert self.get_params()[name] == value
        assert self.use_dynamic_anchors is True
        assert x.patient_id.tolist() == ['saved_patient']
        assert y.tolist() == [1]
        raise StopBeforeTraining

    monkeypatch.setattr(TemporalTree, 'fit', stop)
    with pytest.raises(StopBeforeTraining):
        shared.run_model(True, data, patients, tmp_path, model_config=config)
    effective = json.loads((tmp_path / 'effective_configuration.json').read_text())
    assert effective[name] == value
    assert effective['aggregation_funcs'] == {'sum': 'numpy.sum'}


def test_v2_main_passes_local_configuration_and_same_cohort(monkeypatch, tmp_path):
    from experiments import experiment_2_temporal_explainability as explainability

    source = tmp_path / 'input'
    source.mkdir()
    for name in v2.DATASET_FILES:
        (source / name).write_text('{}')
    data, patients = object(), object()
    monkeypatch.setattr(v2, 'load_dataset', lambda path: (data, patients, {'config': {}}))
    config = deepcopy(v2.MODEL_CONFIG)
    config['max_depth'] = 5
    monkeypatch.setattr(v2, 'MODEL_CONFIG', config)
    calls = []

    def record(enabled, actual_data, actual_patients, output, *, model_config):
        assert actual_data is data and actual_patients is patients
        assert model_config is config
        assert shared.make_model(enabled, model_config).max_depth == 5
        calls.append(enabled)
        return {}

    monkeypatch.setattr(v2, 'run_model', record)
    monkeypatch.setattr(v2, 'save_experiment_outputs', lambda path: None)
    monkeypatch.setattr(explainability, 'save_temporal_reference_usage',
                        lambda path: None)
    output = tmp_path / 'results'
    monkeypatch.setattr('sys.argv', ['runner', '--data-dir', str(source),
                                    '--output', str(output)])
    v2.main()
    assert calls == [True, False]
    saved = json.loads((output / 'configuration.json').read_text())
    assert saved['parameters']['max_depth'] == 5
    assert saved['aggregations'] == ['sum']
