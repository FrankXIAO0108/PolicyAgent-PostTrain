from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_qwen3_4b import build_manifest


class Qwen3SnapshotManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_excludes_cache_and_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "weights.safetensors").write_bytes(b"weights")
            (root / ".cache").mkdir()
            (root / ".cache" / "ignored").write_bytes(b"cache")
            (root / "MODEL_MANIFEST.json").write_text("old", encoding="utf-8")

            first = build_manifest(root, "Qwen/example", "abc123")
            second = build_manifest(root, "Qwen/example", "abc123")

            self.assertEqual(first["aggregate_sha256"], second["aggregate_sha256"])
            expected_bytes = (root / "config.json").stat().st_size + (
                root / "weights.safetensors"
            ).stat().st_size
            self.assertEqual(first["total_bytes"], expected_bytes)
            self.assertEqual(
                [item["path"] for item in first["files"]],
                ["config.json", "weights.safetensors"],
            )


if __name__ == "__main__":
    unittest.main()
