"""Offline cached-score audit against explicit analyst labels, NOT human gold."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from src.evaluation.semantic_shadow_judge import build_packet, digest
from src.evaluation.task44_hybrid_reward import (
    INTERPRETER_REVISION,
    PROMPT,
    score_candidate_response,
)


def error_rates(expected, predicted):
    """Keep missing scores visible; never count abstention as a correct score."""
    if len(expected) != len(predicted):
        raise ValueError("Unpaired labels/scores")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in expected + [p for p in predicted if p is not None]
    ):
        raise ValueError("Invalid score")

    def rate(eligible, wrong):
        denominator = sum(eligible(e) for e in expected)
        count = sum(
            eligible(e) and p is not None and wrong(e, p)
            for e, p in zip(expected, predicted, strict=True)
        )
        missing = sum(
            eligible(e) and p is None for e, p in zip(expected, predicted, strict=True)
        )
        return {
            "count": count,
            "denominator": denominator,
            "rate": count / denominator if denominator else None,
            "unscored": missing,
        }

    def full(x):
        return math.isclose(x, 1.0, rel_tol=0, abs_tol=1e-9)

    return {
        "full_score_wrongly_penalized": rate(full, lambda e, p: p < 1 - 1e-9),
        "non_full_wrongly_given_full": rate(
            lambda e: not full(e), lambda e, p: full(p)
        ),
        "any_score_error": rate(lambda e: True, lambda e, p: abs(e - p) > 1e-9),
        "underrated": rate(lambda e: True, lambda e, p: p < e - 1e-9),
        "overrated": rate(lambda e: True, lambda e, p: p > e + 1e-9),
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(config):
    root = Path(config["source_dir"])
    summary_path = root / "summary.json"
    if sha(summary_path) != config["summary_sha256"]:
        raise ValueError("Summary binding mismatch")
    if (
        config["label_status"] != "CODEX_ANALYST_PROVISIONAL"
        or config["independent_gold"] is not False
    ):
        raise ValueError("This audit is not independent gold")
    source = load(summary_path)
    frozen = load(root / "frozen_prompt.json")
    if hashlib.sha256(frozen["prompt"].encode()).hexdigest() != frozen["sha256"]:
        raise ValueError("Frozen prompt binding mismatch")
    if [r["row_index_zero_based"] for r in config["reviews"]] != list(
        range(len(source["results"]))
    ):
        raise ValueError("Every frozen row needs an explicit review, without filtering")
    rows, bindings = [], {str(summary_path): sha(summary_path)}
    for review, prior in zip(config["reviews"], source["results"], strict=True):
        cache = root / "semantic_cache" / prior["details"]["cache_key"]
        request_path, response_path = cache / "request.json", cache / "response.json"
        for path, field in [
            (request_path, "request_sha256"),
            (response_path, "response_sha256"),
        ]:
            if sha(path) != review[field]:
                raise ValueError("Reviewed evidence binding mismatch")
            bindings[str(path)] = sha(path)
        request, response = load(request_path), load(response_path)
        data, packet = request["scoring_input"], request["packet"]
        original_key = digest(
            {
                "packet": packet,
                "prompt": frozen["prompt"],
                "settings": request["settings"],
            }
        )
        if original_key != cache.name or request["prompt_sha256"] != frozen["sha256"]:
            raise ValueError("Cached request binding mismatch")
        actual_packet = build_packet(data["raw"], packet["policy"], row=0)
        if any(
            packet[k] != actual_packet[k] for k in ["messages", "trajectory_sha256"]
        ):
            raise ValueError("Raw trajectory and packet differ")
        choice = response["choices"][0]
        if (
            choice["finish_reason"] != "stop"
            or response["model"] != request["settings"]["model"]
        ):
            raise ValueError("Incomplete or unexpected model response")
        row = {
            **review,
            "old_reward": prior["hybrid_reward"],
            "new_reward": None,
            "raw_sha256": digest(data["raw"]),
            "response_model": response["model"],
        }
        try:
            result = score_candidate_response(
                data["base"],
                data["raw"],
                data["spec"],
                packet,
                choice["message"]["content"],
            )
            result.update(
                used_as_training_reward=False, scope="OFFLINE_CACHED_REINTERPRETATION"
            )
            row.update(
                status="RESCORED", new_reward=result["offline_reward"], details=result
            )
        except (ValueError, RuntimeError) as exc:
            row.update(status="BLOCKED", error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
    expected = [r["expected_reward"] for r in rows]
    before = [r["old_reward"] for r in rows]
    after = [r["new_reward"] for r in rows]
    return {
        "scope": "DEVELOPMENT_REGRESSION_ON_KNOWN_FAILURES_NOT_HELDOUT_ACCURACY",
        "label_status": config["label_status"],
        "independent_gold": False,
        "new_llm_outputs_obtained": False,
        "external_api_called": False,
        "gpu_used": False,
        "training_release_allowed": False,
        "interpreter_revision": INTERPRETER_REVISION,
        "new_prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "input_sha256": bindings,
        "before": error_rates(expected, before),
        "after": error_rates(expected, after),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite audit")
    report = audit(load(args.reviews))
    report.update(
        command=[sys.executable, *sys.argv],
        config=load(args.reviews),
        reviews_sha256=sha(args.reviews),
        project_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    paths = [
        Path(__file__),
        Path("src/evaluation/task44_claim_roles.py"),
        Path("src/evaluation/task44_hybrid_reward.py"),
        Path("src/evaluation/task44_partial_semantics.py"),
        Path("src/evaluation/task44_reward_evidence.py"),
        Path("src/evaluation/semantic_shadow_judge.py"),
    ]
    report["code_sha256"] = {str(p): sha(p) for p in paths}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps({k: report[k] for k in ["before", "after", "external_api_called"]})
    )


if __name__ == "__main__":
    main()
