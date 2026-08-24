import json
import tempfile
import unittest
from pathlib import Path

from src.training.merge_teacher_sft_adapter import validate_source
from src.training.run_retail_agentic_grpo import directory_sha256, sha256


class MergeTeacherSftAdapterValidationTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        base = root / "base"
        adapter = root / "run" / "teacher_sft_adapter"
        base.mkdir()
        adapter.mkdir(parents=True)
        (base / "model.safetensors").write_bytes(b"base")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        manifest = {
            "schema_version": "retail-teacher-sft-run-v1",
            "scope": "TEACHER_TRAJECTORY_SFT",
            "status": "COMPLETED",
            "git": {"commit": "abc", "dirty_at_start": False},
            "bindings": {
                "config_sha256": "CONFIG",
                "data_manifest_sha256": "DATA",
                "starting_model": str(base),
                "starting_model_sha256": directory_sha256(base),
            },
            "teacher_sft_gate": {"passed": True},
            "artifacts": {
                "adapter": {
                    "path": str(adapter),
                    "sha256": directory_sha256(adapter),
                }
            },
            "business_improvement_claim_allowed": False,
        }
        manifest_path = root / "run" / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return root / "run", manifest_path

    def test_validates_hash_bound_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, manifest_path = self._fixture(Path(directory))
            result = validate_source(run_dir, sha256(manifest_path))
            self.assertEqual(result["source_manifest_sha256"], sha256(manifest_path))
            self.assertEqual(
                result["adapter_sha256"], directory_sha256(run_dir / "teacher_sft_adapter")
            )

    def test_rejects_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, _ = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                validate_source(run_dir, "0" * 64)

    def test_rejects_tampered_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, manifest_path = self._fixture(Path(directory))
            (run_dir / "teacher_sft_adapter" / "new.bin").write_bytes(b"tamper")
            with self.assertRaisesRegex(ValueError, "adapter hash mismatch"):
                validate_source(run_dir, sha256(manifest_path))

    def test_rejects_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, manifest_path = self._fixture(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["teacher_sft_gate"]["passed"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "loss gate did not pass"):
                validate_source(run_dir, sha256(manifest_path))


if __name__ == "__main__":
    unittest.main()
