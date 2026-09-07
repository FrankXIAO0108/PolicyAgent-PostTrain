"""CPU-only witness: real TRL sampler/buffering + Transformers optimizer loop.

Generation and loss are synthetic test doubles. This is not an Agent/RL result.
No model download, tau2 data, API credentials, or GPU is used.
"""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from transformers import Trainer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from src.training.accumulation_contract import align_dataset_for_accumulation, make_accumulation_guard

    torch.set_num_threads(1)
    torch.manual_seed(2026090601)
    settings = {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
                "steps_per_generation": 4, "num_generations": 4, "num_iterations": 1}

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, x):
            return self.weight * x

    class Witness(GRPOTrainer):
        def __init__(self, dataset, output):
            training_args = GRPOConfig(
                output_dir=str(output), max_steps=2, learning_rate=2e-6,
                **settings, use_cpu=True, bf16=False, fp16=False,
                remove_unused_columns=False, report_to="none", save_strategy="no",
                logging_strategy="no", disable_tqdm=True, seed=2026090601,
                dataloader_pin_memory=False, gradient_checkpointing=False,
            )
            Trainer.__init__(self, model=Toy(), args=training_args, train_dataset=dataset,
                             data_collator=lambda rows: rows)
            self.num_generations = 4
            self.num_iterations = 1
            self.shuffle_dataset = True
            self._step = 0
            self._buffered_inputs = None
            self._current_train_step_time = 0.0
            self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
            self.model_accepts_loss_kwargs = False
            self.generated = 0
            self.fixture_rollout_log = Path(output) / "synthetic_rows.jsonl"
            self.consumed = []
            self.updates = []
            owner = self

            class Capture(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    owner.updates.append({"step": state.global_step, "microsteps_total": owner._step,
                                          "generated_total": owner.generated,
                                          "consumed_ids": list(owner.consumed)})

            self.add_callback(Capture())

        def _generate_and_score_completions(self, inputs):
            assert len(inputs) == 4
            ids = torch.arange(self.generated, self.generated + 4)
            self.generated += 4
            with self.fixture_rollout_log.open("a", encoding="utf-8") as stream:
                for row_id in ids.tolist():
                    stream.write(json.dumps({"synthetic_row_id": row_id}) + "\n")
            return {"x": torch.ones(4), "row_id": ids}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            self.consumed.extend(inputs["row_id"].tolist())
            return (model(inputs["x"]) - 1).square().mean()

        log = Trainer.log

    dataset = Dataset.from_list([{"prompt": "SYNTHETIC COUNTER FIXTURE"}])
    old = Witness(dataset, args.output_dir / "old")
    old_loader = len(old.get_train_dataloader())
    old.train()
    aligned, plan = align_dataset_for_accumulation(dataset, settings)
    events = []
    ledger = args.output_dir / "fixed" / "synthetic_rows.jsonl"

    def logged_rows():
        with ledger.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())

    guarded = make_accumulation_guard(Witness, plan, events.append, logged_rows)
    fixed = guarded(aligned, args.output_dir / "fixed")
    fixed_loader = len(fixed.get_train_dataloader())
    fixed.train()
    assert old_loader == 4 and old.generated == 8
    assert [r["microsteps_total"] for r in old.updates] == [4, 8]
    assert fixed_loader == 8 and fixed.generated == 16
    assert [r["microsteps_total"] for r in fixed.updates] == [8, 16]
    assert sorted(fixed.consumed) == list(range(16)), "Each generated row must be consumed once"
    assert sorted(fixed.updates[0]["consumed_ids"]) == list(range(8))
    checks = [e for e in events if e["event"] == "before_optimizer_step"]
    assert len(checks) == 2 and all(e["status"] == "PASS" for e in checks)
    assert [e["logged_rollouts_total"] for e in checks] == [8, 16]
    blocked = guarded(dataset, args.output_dir / "blocked")
    try:
        blocked.train()
    except RuntimeError as exc:
        assert "partial accumulation" in str(exc)
        assert blocked.generated == 0 and blocked.state.global_step == 0
        blocked_reason = str(exc)
    else:
        raise AssertionError("Unaligned loader was not blocked before generation")
    report = {
        "status": "PASS", "scope": "CPU synthetic iteration witness; NOT an RL experiment",
        "external_api_called": False, "gpu_used": False,
        "versions": {"torch": torch.__version__, "trl": trl.__version__, "transformers": transformers.__version__},
        "source_hashes": {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                          for cls in (GRPOTrainer, Trainer)},
        "alignment": plan, "old": {"loader_length": old_loader, "updates": old.updates},
        "fixed": {"loader_length": fixed_loader, "updates": fixed.updates, "events": events},
        "unaligned_rejected_before_generation": blocked_reason,
    }
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
