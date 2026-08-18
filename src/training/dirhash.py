"""Canonical directory digest used to bind experiment artifacts.

Files are walked in a platform-independent order: relative posix paths
sorted as plain strings (case-sensitive). Sorting raw pathlib.Path objects
is NOT deterministic across platforms because WindowsPath compares
case-insensitively while PosixPath compares case-sensitively; that mismatch
made the same directory hash differently on Linux vs Windows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in _files_in_canonical_order(path):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


def _files_in_canonical_order(path: Path) -> Iterator[Path]:
    yield from sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    )
