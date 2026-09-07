"""Single-process, fresh-rollout GRPO accumulation; no reward or loss changes."""
from math import gcd


def align_dataset_for_accumulation(dataset, grpo):
    """Tile the entire prompt pool equally; never pad only a subset of tasks."""
    size = len(dataset)
    micro = int(grpo["per_device_train_batch_size"])
    accumulation = int(grpo["gradient_accumulation_steps"])
    generation_steps = int(grpo["steps_per_generation"])
    n = int(grpo["num_generations"])
    if min(size, micro, accumulation, generation_steps, n) <= 0:
        raise ValueError("Accumulation alignment requires positive sizes")
    if int(grpo.get("num_iterations", 1)) != 1:
        raise ValueError("Accumulation alignment supports fresh rollouts only")
    if accumulation % generation_steps or (micro * generation_steps) % n:
        raise ValueError("Accumulation requires complete generation batches")
    prompts_per_update = micro * accumulation // n
    repeats = prompts_per_update // gcd(size, prompts_per_update)
    aligned = dataset.select(list(range(size)) * repeats) if repeats > 1 else dataset
    plan = {
        "schema_version": "grpo-epoch-alignment-v1",
        "source_prompt_rows": size,
        "dataset_repetitions": repeats,
        "training_prompt_rows": len(aligned),
        "expected_microsteps_per_epoch": len(aligned) * n // micro,
        "microsteps_per_update": accumulation,
        "fresh_rollouts_per_update": micro * accumulation,
        "reward_groups_per_update": prompts_per_update,
        "group_size": n,
        "source_fingerprint": getattr(dataset, "_fingerprint", None),
        "aligned_fingerprint": getattr(aligned, "_fingerprint", None),
        "world_size": 1,
    }
    return aligned, plan


def make_accumulation_guard(base, plan, emit, rollout_count=None):
    """Check the real loader and actual counts before each optimizer update."""
    from transformers import TrainerCallback

    class Guarded(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._accum_microsteps = 0
            self._accum_fresh_rollouts = 0
            self._accum_generation_batches = 0
            self._accum_previous = (0, 0, 0)
            trainer = self

            class CheckUpdate(TrainerCallback):
                def on_pre_optimizer_step(self, args, state, control, **kwargs):
                    current = (trainer._accum_microsteps, trainer._accum_fresh_rollouts,
                               trainer._accum_generation_batches)
                    delta = tuple(a - b for a, b in zip(current, trainer._accum_previous))
                    expected = (plan["microsteps_per_update"], plan["fresh_rollouts_per_update"],
                                args.gradient_accumulation_steps // args.steps_per_generation)
                    logged = rollout_count() if rollout_count is not None else None
                    passed = delta == expected and (logged is None or logged == current[1])
                    emit({"event": "before_optimizer_step", "status": "PASS" if passed else "FAIL",
                          "optimizer_step": state.global_step + 1, "microsteps": delta[0],
                          "fresh_rollouts": delta[1], "generation_batches": delta[2],
                          "logged_rollouts_total": logged})
                    if not passed:
                        raise RuntimeError(f"Accumulation count mismatch before update: {delta} != {expected}")
                    trainer._accum_previous = current

            self.add_callback(CheckUpdate())

        def get_train_dataloader(self):
            if self.accelerator.num_processes != 1:
                raise RuntimeError("Audited accumulation contract supports one process only")
            loader = super().get_train_dataloader()
            count = len(loader)
            if (count != plan["expected_microsteps_per_epoch"]
                    or count % plan["microsteps_per_update"]):
                raise RuntimeError(f"Epoch would flush a partial accumulation window: {count} microsteps")
            emit({"event": "dataloader_validated", "microsteps_per_epoch": count})
            return loader

        def _generate_and_score_completions(self, inputs):
            was_training = self.model.training
            result = super()._generate_and_score_completions(inputs)
            if was_training:
                self._accum_fresh_rollouts += len(inputs)
                self._accum_generation_batches += 1
            return result

        def training_step(self, model, inputs, num_items_in_batch):
            result = super().training_step(model, inputs, num_items_in_batch)
            self._accum_microsteps += 1
            return result

    return Guarded
