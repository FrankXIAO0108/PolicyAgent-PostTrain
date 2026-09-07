"""CPU activation/lifecycle contract tests; not Qwen/NF4 GPU training evidence."""

from threading import Event, Thread
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from src.training import run_retail_agentic_grpo as runner
from src.training.rollout_diagnostics import (
    BudgetTrace,
    GuardedTrajectoryTrace,
    _exclusive_sampling_model,
    make_guarded_grpo_trainer,
    make_sampling_trainer,
    qwen3_nf4_bf16_activation_context,
)

torch = pytest.importorskip("torch")


def test_trainable_parameter_fingerprint_detects_real_finite_update():
    model = torch.nn.Linear(3, 2, bias=False)
    before = runner.trainable_parameter_fingerprint(model)
    with torch.no_grad():
        model.weight.add_(0.25)
    after = runner.trainable_parameter_fingerprint(model)
    assert before["all_finite"] and after["all_finite"]
    assert before["sha256"] != after["sha256"]
    assert before["tensor_count"] == after["tensor_count"] == 1
    comparison = runner.compare_trainable_fingerprints(before, after)
    assert comparison["tensor_keys_match"]
    assert comparison["tensor_metadata_match"]
    assert comparison["changed_tensor_count"] == 1
    assert comparison["parameter_change_detected"]


def test_trainable_parameter_fingerprint_reports_nonfinite_values():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
        model.weight[0, 1] = float("inf")
    result = runner.trainable_parameter_fingerprint(model)
    assert result["all_finite"] is False
    assert result["nonfinite_count"] == 2


def test_training_log_evidence_requires_real_finite_kl_rows():
    missing = runner.summarize_training_log(
        [{"loss": 0.2, "reward_std": 0.1, "grad_norm": 0.3}]
    )
    assert missing["kl_curve_available"] is False
    present = runner.summarize_training_log(
        [{"loss": 0.2, "reward_std": 0.1, "grad_norm": 0.3, "kl": 0.02}]
    )
    assert present["kl_curve_available"] is True
    nonfinite = runner.summarize_training_log([{"kl": float("nan")}])
    assert nonfinite["kl_curve_available"] is False
    assert nonfinite["all_monitored_values_finite"] is False
    assert nonfinite["series"]["kl"] == []


def test_training_log_evidence_requires_an_actual_finite_grad_norm():
    missing = runner.summarize_training_log(
        [{"loss": 0.2, "reward_std": 0.1, "kl": 0.02}]
    )
    assert missing["finite_gradients_recorded"] is False

    present = runner.summarize_training_log(
        [{"loss": 0.2, "reward_std": 0.1, "grad_norm": 0.3, "kl": 0.02}]
    )
    assert present["finite_gradients_recorded"] is True

    nonfinite = runner.summarize_training_log(
        [{"loss": 0.2, "reward_std": 0.1, "grad_norm": float("nan")}]
    )
    assert nonfinite["finite_gradients_recorded"] is False
    assert nonfinite["all_monitored_values_finite"] is False


class Qwen3RMSNorm(torch.nn.Module):
    """Small FP32-output double of the relevant norm output contract."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, value):
        value = value.float()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * self.weight
        )


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.is_loaded_in_4bit = True  # Test double only, no quantized GPU backend.
        self.embedding = torch.nn.Embedding(8, 4)
        self.norm = Qwen3RMSNorm()
        self.head = torch.nn.Linear(4, 5, bias=False)
        self.observed = []

    def forward(self, ids):
        embedded = self.embedding(ids)
        normalized = self.norm(embedded)
        self.observed.append((embedded.dtype, normalized.dtype))
        return self.head(normalized.float())


class PeftWrapper(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def get_base_model(self):
        return self.core


class ModuleWrapper(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.module = core


def assert_no_hooks(core):
    assert all(not module._forward_hooks for module in core.modules())


def test_whole_lifetime_preserves_gradients_and_actual_optimizer_update():
    actor, reference = Core(), Core()
    actor.embedding.weight.requires_grad_(False)
    reference.requires_grad_(False)
    actor.eval()  # Context must not silently change caller modes.
    before = {
        name: parameter.detach().clone() for name, parameter in actor.named_parameters()
    }
    flags = [(p.requires_grad, p.dtype) for p in actor.parameters()]
    optimizer = torch.optim.SGD(
        (p for p in actor.parameters() if p.requires_grad), lr=0.05
    )
    events = []
    ids = torch.tensor([[1, 2]])
    with qwen3_nf4_bf16_activation_context(
        [ModuleWrapper(PeftWrapper(actor)), reference], events.append
    ):
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        assert not actor.training
        assert flags == [(p.requires_grad, p.dtype) for p in actor.parameters()]
        assert all(torch.equal(before[name], p) for name, p in actor.named_parameters())
        with torch.no_grad():
            actor(ids)  # Generation/old-policy scoring use the same installed hooks.
            reference(ids)
            assert not torch.is_grad_enabled()
        output = actor(ids)  # Current policy scoring, loss, backward and optimizer.
        loss = torch.nn.functional.cross_entropy(
            output.reshape(-1, 5), torch.tensor([3, 4])
        )
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in actor.parameters()
            if p.requires_grad
        )
        assert actor.head.weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before["head.weight"], actor.head.weight)
        assert all(
            pair == (torch.bfloat16, torch.bfloat16)
            for pair in actor.observed + reference.observed
        )
    assert_no_hooks(actor)
    assert_no_hooks(reference)
    assert events[0]["core_count"] == 2
    assert events[-1]["hooks_removed"] == 4
    assert events[-1]["status"] == "COMPLETED"
    assert flags == [(p.requires_grad, p.dtype) for p in actor.parameters()]


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_recomputation_keeps_bf16_outputs_until_backward_finishes(
    use_reentrant,
):
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    core = Core()
    seen = []
    value = torch.randn(2, 4, requires_grad=True)
    with qwen3_nf4_bf16_activation_context([core], lambda event: None):
        observer = core.norm.register_forward_hook(
            lambda module, args, output: seen.append(output.dtype)
        )
        try:
            with set_checkpoint_early_stop(False):
                output = checkpoint(core.norm, value, use_reentrant=use_reentrant)
            assert output.dtype == torch.bfloat16
            output.float().square().sum().backward()
        finally:
            observer.remove()
        assert len(seen) >= 2 and set(seen) == {torch.bfloat16}
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert (
            core.norm.weight.grad is not None
            and torch.isfinite(core.norm.weight.grad).all()
        )
    assert_no_hooks(core)


def test_duplicate_reference_core_is_not_hooked_twice_and_generation_lock_is_independent():
    core = Core()
    events = []
    with qwen3_nf4_bf16_activation_context(
        [core, PeftWrapper(core), ModuleWrapper(core), None], events.append
    ) as details:
        assert details["core_count"] == 1
        assert details["embedding_modules"] == details["rmsnorm_modules"] == 1
        assert len(core.norm._forward_hooks) == len(core.embedding._forward_hooks) == 1
        with _exclusive_sampling_model(core):
            assert core.norm(torch.ones(2, 4)).dtype == torch.bfloat16
        with pytest.raises(RuntimeError, match="Concurrent or nested"):
            with qwen3_nf4_bf16_activation_context([PeftWrapper(core)], events.append):
                pytest.fail("Nested scope must not be entered")
        assert len(core.norm._forward_hooks) == 1
    assert_no_hooks(core)


@pytest.mark.parametrize(
    "kind", ["wrong_type", "not_4bit", "missing_norm", "missing_embedding", "empty"]
)
def test_unsupported_models_rejected_before_any_hook_is_installed(kind):
    good, bad = Core(), Core()
    if kind == "wrong_type":
        bad.config.model_type = "qwen2"
    elif kind == "not_4bit":
        bad.is_loaded_in_4bit = False
    elif kind == "missing_norm":
        bad.norm = torch.nn.Identity()
    elif kind == "missing_embedding":
        bad.embedding = torch.nn.Identity()
    with pytest.raises(ValueError):
        with qwen3_nf4_bf16_activation_context(
            [] if kind == "empty" else [good, bad], lambda event: None
        ):
            pytest.fail("Unsupported context must not be entered")
    assert_no_hooks(good)
    assert_no_hooks(bad)


def test_body_error_cleans_hooks_and_releases_ownership_without_masking_error():
    core = Core()
    events = []
    with pytest.raises(RuntimeError, match="body failure"):
        with qwen3_nf4_bf16_activation_context([core], events.append):
            raise RuntimeError("body failure")
    assert_no_hooks(core)
    assert events[-1]["status"] == "ERROR"
    assert core.norm(torch.ones(1, 4)).dtype == torch.float32
    with qwen3_nf4_bf16_activation_context([core], events.append):
        pass


def test_partial_hook_installation_failure_cleans_earlier_hooks(monkeypatch):
    core = Core()
    original = core.norm.register_forward_hook

    def failed_install(*args, **kwargs):
        raise RuntimeError("hook installation failed")

    monkeypatch.setattr(core.norm, "register_forward_hook", failed_install)
    with pytest.raises(RuntimeError, match="hook installation failed"):
        with qwen3_nf4_bf16_activation_context([core], lambda event: None):
            pass
    assert_no_hooks(core)
    monkeypatch.setattr(core.norm, "register_forward_hook", original)
    with qwen3_nf4_bf16_activation_context([core], lambda event: None):
        pass


@pytest.mark.parametrize("phase", ["enter", "exit"])
def test_logging_failure_still_cleans_and_releases(phase):
    core = Core()

    def emit(event):
        if event["event"] == f"training_precision_{phase}":
            raise OSError("logging unavailable")

    with pytest.raises(OSError, match="logging unavailable"):
        with qwen3_nf4_bf16_activation_context([core], emit):
            pass
    assert_no_hooks(core)
    with qwen3_nf4_bf16_activation_context([core], lambda event: None):
        pass


def test_concurrent_context_cannot_mutate_an_already_claimed_core():
    core = Core()
    entered, release = Event(), Event()
    failures = []

    def hold():
        try:
            with qwen3_nf4_bf16_activation_context([core], lambda event: None):
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test release not received")
        except BaseException as exc:
            failures.append(exc)

    thread = Thread(target=hold)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(RuntimeError, match="Concurrent or nested"):
            with qwen3_nf4_bf16_activation_context(
                [PeftWrapper(core)], lambda event: None
            ):
                pytest.fail("Concurrent scope must not enter")
        assert len(core.norm._forward_hooks) == 1
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not failures
    assert_no_hooks(core)


@pytest.mark.parametrize("factory", ["optimize", "sample"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_factory_bf16_flag_enters_only_native_generation_and_always_exits(
    monkeypatch, factory, enabled, raise_inside
):
    calls = []
    output = ([[7, 9], [8, 9]], None)

    @contextmanager
    def autocast():
        calls.append("amp_enter")
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                yield
        finally:
            calls.append("amp_exit")

    class Native:
        def __init__(self):
            self.model = Core()
            self.accelerator = SimpleNamespace(num_processes=1, autocast=autocast)
            self.use_vllm = self._is_vlm = self.use_transformers_continuous_batching = (
                False
            )
            self._budget_trace = (
                GuardedTrajectoryTrace(8, 100, lambda event: None)
                if factory == "sample"
                else BudgetTrace(8, 100, lambda event: None)
            )
            self._tokenizer = SimpleNamespace(eos_token_id=9)
            self.generation_config = SimpleNamespace(max_new_tokens=8)
            self.generation_kwargs = {}

        def _generate_single_turn(self, prompts, images, fields):
            calls.append("native")
            assert torch.is_autocast_enabled("cpu") is enabled
            assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            if raise_inside:
                raise RuntimeError("native generation failed")
            return output

    def generation_guard(
        core, prompts, indices, generation_index, emit, native, images, fields,
        expected_max_new_tokens=None,
    ):
        assert expected_max_new_tokens == (8 if factory == "optimize" else None)
        assert not torch.is_autocast_enabled("cpu")
        try:
            return native(prompts, images, fields)
        finally:
            assert not torch.is_autocast_enabled("cpu")

    monkeypatch.setattr(
        "src.training.rollout_diagnostics._generate_with_finished_row_guard",
        generation_guard,
    )
    cls = (
        make_guarded_grpo_trainer(Native, lambda event: None, bf16_generation=enabled)
        if factory == "optimize"
        else make_sampling_trainer(Native, bf16_generation=enabled)
    )
    trainer = cls()
    if raise_inside:
        with pytest.raises(RuntimeError, match="native generation failed"):
            trainer._generate_single_turn([[1, 2], [3, 4]], None, None)
    else:
        result = trainer._generate_single_turn([[1, 2], [3, 4]], None, None)
        assert result[0] is output[0] and result[1] is output[1]
    assert calls == (["amp_enter", "native", "amp_exit"] if enabled else ["native"])
    assert not torch.is_autocast_enabled("cpu")


def test_cuda_flag_without_real_bf16_autocast_fails_before_native(monkeypatch):
    from src.training.rollout_diagnostics import _bf16_native_generation

    calls = []

    @contextmanager
    def autocast():
        try:
            yield
        finally:
            calls.append("exit")

    core = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))])
    )
    trainer = SimpleNamespace(accelerator=SimpleNamespace(autocast=autocast))
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: False)
    wrapped = _bf16_native_generation(trainer, core, lambda: calls.append("native"))
    with pytest.raises(RuntimeError, match="actual BF16"):
        wrapped()
    assert calls == ["exit"]
