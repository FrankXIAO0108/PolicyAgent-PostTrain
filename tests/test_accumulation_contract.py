"""No model downloads or API: epoch alignment and pre-update accounting."""
from types import SimpleNamespace

import pytest

from src.training.accumulation_contract import align_dataset_for_accumulation, make_accumulation_guard


class Rows(list):
    def select(self, indices):
        return Rows(self[i] for i in indices)


def settings():
    return {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
            "steps_per_generation": 4, "num_generations": 4, "num_iterations": 1}


@pytest.mark.parametrize("size,repeats,microsteps", [(1, 2, 8), (2, 1, 8), (3, 2, 24), (4, 1, 16)])
def test_whole_pool_tiling_preserves_equal_task_weight(size, repeats, microsteps):
    data = Rows({"task_id": str(i), "prompt": str(i)} for i in range(size))
    aligned, plan = align_dataset_for_accumulation(data, settings())
    assert plan["dataset_repetitions"] == repeats
    assert plan["expected_microsteps_per_epoch"] == microsteps
    assert aligned == data * repeats
    assert plan["fresh_rollouts_per_update"] == 8
    assert plan["reward_groups_per_update"] == 2
    assert plan["group_size"] == 4


@pytest.mark.parametrize("changes", [{"num_iterations": 2}, {"gradient_accumulation_steps": 6},
                                     {"num_generations": 3}, {"steps_per_generation": 0}])
def test_unsupported_shapes_fail(changes):
    with pytest.raises(ValueError):
        align_dataset_for_accumulation(Rows([1]), {**settings(), **changes})


def fake_trainer(loader_size=8, world_size=1, logged=None):
    class Base:
        def __init__(self):
            self.model = SimpleNamespace(training=True)
            self.accelerator = SimpleNamespace(num_processes=world_size)

        def add_callback(self, callback):
            self.callback = callback

        def get_train_dataloader(self):
            return range(loader_size)

        def _generate_and_score_completions(self, inputs):
            return inputs

        def training_step(self, model, inputs, num_items_in_batch):
            return 1

    _, plan = align_dataset_for_accumulation(Rows([1]), settings())
    events = []
    return make_accumulation_guard(Base, plan, events.append, logged)(), events


@pytest.mark.parametrize("size,world", [(4, 1), (12, 1), (8, 2)])
def test_real_loader_boundary_or_distributed_change_rejected(size, world):
    trainer, _ = fake_trainer(size, world)
    with pytest.raises(RuntimeError):
        trainer.get_train_dataloader()


@pytest.mark.parametrize("microsteps,groups,logged", [(4, 1, 4), (8, 1, 4), (8, 2, 7)])
def test_bad_update_rejected_before_optimizer(microsteps, groups, logged):
    trainer, events = fake_trainer(logged=lambda: logged)
    for _ in range(groups):
        trainer._generate_and_score_completions([1] * 4)
    for _ in range(microsteps):
        trainer.training_step(None, None, None)
    with pytest.raises(RuntimeError):
        trainer.callback.on_pre_optimizer_step(SimpleNamespace(**settings()), SimpleNamespace(global_step=0), None)
    assert events[-1]["status"] == "FAIL"


def test_two_separate_groups_then_one_update_repeated():
    trainer, events = fake_trainer()
    assert len(trainer.get_train_dataloader()) == 8
    for step in range(2):
        for _ in range(2):
            trainer._generate_and_score_completions([1] * 4)
            for _ in range(4):
                trainer.training_step(None, None, None)
        trainer.callback.on_pre_optimizer_step(SimpleNamespace(**settings()), SimpleNamespace(global_step=step), None)
    assert [e["microsteps"] for e in events[1:]] == [8, 8]
    assert [e["fresh_rollouts"] for e in events[1:]] == [8, 8]
    assert [e["generation_batches"] for e in events[1:]] == [2, 2]
