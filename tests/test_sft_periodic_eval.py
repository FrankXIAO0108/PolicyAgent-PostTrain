import pytest
from src.training.run_teacher_sft import periodic_evaluation_args


def test_old_runs_unchanged():
    assert periodic_evaluation_args({'max_steps': 20}) == {}


def test_fifty_step_selection():
    args = periodic_evaluation_args({'max_steps': 50, 'eval_steps': 10})
    assert args['eval_steps'] == 10
    assert args['load_best_model_at_end']
    assert args['metric_for_best_model'] == 'eval_validation_loss'
    assert args['prediction_loss_only']


@pytest.mark.parametrize('interval', [0, -1, 3, True, 2.5])
def test_bad_interval(interval):
    with pytest.raises(ValueError):
        periodic_evaluation_args({'max_steps': 50, 'eval_steps': interval})
