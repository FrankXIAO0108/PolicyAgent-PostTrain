from __future__ import annotations

import unittest

from src.training.run_scripted_replay import remap_path


class ScriptedReplayPathRemapTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
