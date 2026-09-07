"""Offline real-runtime budget probe; random tiny weights, no business rollout."""

import pytest


def test_real_trl_remaining_budget_stops_without_fabricated_eos(tmp_path):
    pytest.importorskip("trl")
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    from src.training.rollout_diagnostics import (
        GuardedTrajectoryTrace,
        make_guarded_grpo_trainer,
        verify_trl_source,
    )

    verify_trl_source(GRPOTrainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(
            {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "a": 3}, unk_token="[UNK]"
        )),
        pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
    )
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=4, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=1, tie_word_embeddings=False,
    ))
    # Deterministic argmax is PAD, never EOS: only the length cap can stop it.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    args = GRPOConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        report_to="none", max_steps=1, num_generations=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
        steps_per_generation=1, max_completion_length=8, beta=0.0,
        generation_kwargs={"do_sample": False}, save_strategy="no",
    )
    events = []
    trainer = make_guarded_grpo_trainer(GRPOTrainer, events.append)(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list([{"prompt": "a"}]),
        reward_funcs=lambda **kw: pytest.fail("Probe must never calculate reward"),
    )
    trace = GuardedTrajectoryTrace(8, 128, events.append)
    trace.start([[3], [3]])
    # A prior model turn plus observations have used five tokens in each row.
    prompts = [[3] * 6, [3] * 6]
    trace.next_generation = list(enumerate(prompts))
    trainer._policyagent_stop_trace = trace
    original = trainer.generation_config
    try:
        ids, _ = trainer._generate_single_turn(prompts, None, {})
    finally:
        del trainer._policyagent_stop_trace
    assert [len(row) for row in ids] == [3, 3]
    assert all(tokenizer.eos_token_id not in row for row in ids)
    assert trainer.generation_config is original
    assert all("COMPLETION_BUDGET_EXHAUSTED" in row["flags"] for row in trace.rows)
    assert trainer.state.global_step == 0
    assert trainer.optimizer is None
    assert next(model.parameters()).device.type == "cpu"
    assert any(e["event"] == "remaining_generation_budget" and
               e["max_new_tokens"] == 3 for e in events)
