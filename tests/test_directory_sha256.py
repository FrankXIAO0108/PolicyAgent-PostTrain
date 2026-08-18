import hashlib
import tempfile
import unittest
from pathlib import Path

from src.training.dirhash import directory_sha256


def _expected(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[name])
        digest.update(b"\0")
    return digest.hexdigest().upper()


class DirectorySha256Tests(unittest.TestCase):
    def test_matches_canonical_order_regardless_of_platform(self):
        files = {"adapter_config.json": b'{"r": 16}\n', "README.md": b"# readme\n", "tokenizer.json": b"{}\n"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in files.items():
                (root / name).write_bytes(content)
            self.assertEqual(directory_sha256(root), _expected(files))

    def test_mixed_case_names_use_case_sensitive_posix_order(self):
        # WindowsPath sorts case-insensitively, PosixPath case-sensitively.
        # The canonical digest must always place README.md before adapter_*.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_config.json").write_bytes(b"a")
            (root / "README.md").write_bytes(b"r")
            self.assertEqual(
                directory_sha256(root),
                _expected({"README.md": b"r", "adapter_config.json": b"a"}),
            )

    def test_nested_directory_uses_relative_posix_paths(self):
        files = {"sub/README.md": b"x", "adapter_config.json": b"y"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            for name, content in files.items():
                (root / name).write_bytes(content)
            self.assertEqual(directory_sha256(root), _expected(files))

    def test_empty_directory_hashes_empty_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                directory_sha256(Path(tmp)),
                hashlib.sha256(b"").hexdigest().upper(),
            )


if __name__ == "__main__":
    unittest.main()
