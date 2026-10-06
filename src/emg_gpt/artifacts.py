"""Artifact identities and optional downloads, independent of training provenance."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path


def resource_json(name: str) -> dict:
    return json.loads(files("emg_gpt").joinpath("resources", name).read_text())


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(path: Path, identity: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
        raise ValueError(f"Artifact SHA-256 or size mismatch: {path.name}")


def fetch_tokenizer(destination: Path | str) -> Path:
    """Reuse or download the pinned upstream tokenizer; never replace a mismatched file."""
    destination = Path(destination)
    artifact = resource_json("artifacts.json")["tokenizer"]
    if destination.exists():
        verify_file(destination, artifact)
        return destination
    source = artifact["public_source"]
    if re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None:
        raise ValueError("Tokenizer download is not pinned to an immutable revision")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Install emg-gpt[download] to fetch weights; local loading needs no Hub client"
        ) from error
    cached = Path(
        hf_hub_download(
            repo_id=source["repo_id"], revision=source["revision"], filename=source["filename"]
        )
    )
    verify_file(cached, artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        staged = Path(handle.name)
    try:
        shutil.copyfile(cached, staged)
        verify_file(staged, artifact)
        os.link(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return destination
