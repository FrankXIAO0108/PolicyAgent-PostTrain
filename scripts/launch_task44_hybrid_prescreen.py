"""Scoped remote launcher: only one n=4 Task44 sampling run, never training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    config = root / "configs/retail_agentic_qwen3_4b_task44_hybrid_prescreen_n4_v1.json"
    cfg = json.loads(config.read_text())
    assert cfg["execution_mode"] == "ROLLOUT_DIAGNOSTIC"
    assert cfg["data"]["task_ids"] == ["44"]
    assert cfg["diagnostic"]["expected_rollouts"] == 4
    assert cfg["grpo"]["num_generations"] == 4
    assert cfg["grpo"]["max_completion_length"] == 8192
    assert cfg["claims"]["parameter_update_allowed"] is False
    bindings = json.loads((root / "upload_manifest.json").read_text())
    for item in bindings["files"]:
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    env = dict(os.environ)
    env.update(POLICYAGENT_TAU2_ROOT="/root/autodl-tmp/tau2-bench", PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1", POLICYAGENT_USER_MODEL="deepseek/deepseek-chat",
               POLICYAGENT_USER_LLM_ARGS_JSON="{}", SHADOW_JUDGE_BASE_URL="https://api.deepseek.com")
    command = [sys.executable, "-m", "src.training.run_retail_agentic_grpo",
               "--config", str(config), "--allow-dirty"]
    if not args.launch:
        with (root / "preflight.log").open("x") as output:
            checked = subprocess.run(command + ["--preflight-only"], env=env, stdout=output, stderr=subprocess.STDOUT)
        if checked.returncode:
            raise SystemExit(checked.returncode)
        with (root / "preflight_passed.json").open("x") as output:
            json.dump({"status": "PASSED", "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                       "external_api_called": False, "command": command + ["--preflight-only"]}, output, indent=2)
        print("PREFLIGHT_PASSED")
        return
    passed = json.loads((root / "preflight_passed.json").read_text())
    assert passed["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert env.get("DEEPSEEK_API_KEY"), "Missing remote simulator credential"
    env["SHADOW_JUDGE_API_KEY"] = env["DEEPSEEK_API_KEY"]
    command += ["--sample-only", "--completion-budget", "8192", "--groups-per-task", "1",
                "--output-dir", str(root / "run")]
    assert not (root / "run").exists()
    with (root / "launch.json").open("x") as output:
        json.dump({"command": command, "created_unix": time.time(), "training_authorized": False,
                   "maximum_agent_rollouts": 4, "maximum_semantic_requests": 4,
                   "user_model": env["POLICYAGENT_USER_MODEL"], "user_llm_args": {},
                   "secret_contents_persisted": False}, output, indent=2)
    with (root / "console.log").open("x") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    with (root / "pid.json").open("x") as output:
        json.dump({"pid": proc.pid}, output)
    print(json.dumps({"status": "STARTED", "pid": proc.pid, "output_dir": str(root / "run")}))


if __name__ == "__main__":
    main()
