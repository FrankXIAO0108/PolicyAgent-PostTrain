from __future__ import annotations

import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from src.training.run_scripted_replay import (
    REPLAY_EVALUATION_TYPE,
    _require_replay_reward,
    remap_path,
    run,
)


class ScriptedReplayPathRemapTests(unittest.TestCase):
    def test_replay_uses_diagnostic_evaluation_without_nl_judge(self) -> None:
        self.assertEqual(REPLAY_EVALUATION_TYPE, "ALL_IGNORE_BASIS")

    def test_missing_reward_info_is_reported_as_infrastructure_failure(self) -> None:
        simulation = SimpleNamespace(
            reward_info=None,
            termination_reason="evaluation_failed",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "completed without reward_info.*evaluation_failed",
        ):
            _require_replay_reward(simulation)

    def test_windows_source_path_maps_to_posix_with_normalized_separators(self) -> None:
        raw = (
            r"D:\PolicyAgent-PostTrain\_local_private_runs"
            r"\correction_targets_20260821\manifest.json"
        )

        mapped = remap_path(
            raw,
            [
                (
                    r"D:\PolicyAgent-PostTrain",
                    "/root/autodl-tmp/PolicyAgent-PostTrain",
                )
            ],
        )

        self.assertEqual(
            mapped,
            "/root/autodl-tmp/PolicyAgent-PostTrain/"
            "_local_private_runs/correction_targets_20260821/manifest.json",
        )

    def test_unmatched_path_is_unchanged(self) -> None:
        raw = r"D:\tau2-bench\data\tau2\domains\retail\policy.md"

        self.assertEqual(
            remap_path(raw, [(r"D:\PolicyAgent-PostTrain", "/root/project")]),
            raw,
        )

    @patch("src.training.run_retail_agentic_grpo.validate_upstream_checkout")
    @patch("src.training.run_scripted_replay.git_value")
    def test_transferred_upstream_package_hash_is_forwarded(
        self,
        git_value_mock,
        validate_upstream_mock,
    ) -> None:
        git_value_mock.side_effect = lambda _root, *args: (
            "" if args == ("status", "--porcelain") else "test-value"
        )
        validate_upstream_mock.return_value = {
            "commit": "upstream-commit",
            "verification_method": "commit_marker_and_source_package_sha256",
        }
        validated = {
            "spec_dir": "/tmp/specs",
            "manifest_sha256_lf": "manifest-hash",
            "seed_source": 20260818,
            "derived_seed": 350291,
            "upstream_commit": "upstream-commit",
            "specs": [],
        }

        with TemporaryDirectory() as temp_dir:
            manifest = run(
                validated,
                Path(temp_dir) / "output",
                20260818,
                "deepseek/deepseek-chat",
                False,
                "SOURCE-PACKAGE-HASH",
                ["python", "-m", "src.training.run_scripted_replay"],
            )

        validate_upstream_mock.assert_called_once_with(
            "upstream-commit",
            expected_package_sha256="SOURCE-PACKAGE-HASH",
        )
        self.assertEqual(
            manifest["evaluation"]["type"],
            "ALL_IGNORE_BASIS",
        )
        self.assertEqual(
            manifest["command"],
            ["python", "-m", "src.training.run_scripted_replay"],
        )


if __name__ == "__main__":
    unittest.main()
