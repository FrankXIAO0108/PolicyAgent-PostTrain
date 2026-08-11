from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = Path("/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_manifest(
    output_dir: Path, model_id: str, revision: str
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    candidates = (
        candidate for candidate in output_dir.rglob("*") if candidate.is_file()
    )
    for path in sorted(candidates):
        relative = path.relative_to(output_dir).as_posix()
        if relative == "MODEL_MANIFEST.json" or relative.startswith(".cache/"):
            continue
        file_hash = sha256(path)
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": file_hash})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\0")
    return {
        "schema_version": "policyagent-model-snapshot-v1",
        "model_id": model_id,
        "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "aggregate_sha256": aggregate.hexdigest().upper(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info(args.model_id).sha
    snapshot_download(
        repo_id=args.model_id,
        revision=revision,
        local_dir=output_dir,
    )
    manifest = build_manifest(output_dir, args.model_id, revision)
    manifest_path = output_dir / "MODEL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
