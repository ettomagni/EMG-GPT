"""Check source hygiene, artifact identities and citation metadata."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


def check(root: Path) -> list[str]:
    errors = []
    required = [
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "MANIFEST.in",
        "docs/inference.md",
        "images/architecture.png",
        "src/emg_gpt/resources/artifacts.json",
        "src/emg_gpt/resources/tokenizer.json",
        ".github/workflows/ci.yml",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not any(x in IGNORE or x.endswith(".egg-info") for x in p.relative_to(root).parts)
    ]
    forbidden = {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".hdf5",
        ".h5",
        ".npy",
        ".npz",
        ".ipynb",
        ".sbatch",
        ".zip",
        ".gz",
        ".pyc",
    }
    patterns = {
        "local absolute path": re.compile(r"/(?:Users|home|scratch)/[A-Za-z0-9_.-]+/"),
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "credential": re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b"
        ),
    }
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.is_symlink() or path.suffix in forbidden or path.name.startswith(".env"):
            errors.append(f"Unwanted release file: {name}")
            continue
        if name == "images/architecture.png":
            if path.stat().st_size > 2_000_000 or not path.read_bytes().startswith(
                b"\x89PNG\r\n\x1a\n"
            ):
                errors.append("Architecture figure must be a PNG smaller than 2 MB")
            continue
        if path.stat().st_size > 1_000_000:
            errors.append(f"Unexpected large source file: {name}")
        try:
            text = path.read_text()
        except UnicodeError:
            errors.append(f"Unexpected binary source file: {name}")
            continue
        for label, pattern in patterns.items():
            if pattern.search(text):
                errors.append(f"Possible {label}: {name}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=name)
            except SyntaxError:
                errors.append(f"Invalid Python syntax: {name}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if (
                    "://" not in target
                    and not target.startswith("#")
                    and not (path.parent / target.split("#")[0]).exists()
                ):
                    errors.append(f"Broken local link: {name} -> {target}")
    if len(list((root / "licenses").glob("*.txt"))) != 5:
        errors.append("Expected five third-party texts plus LICENSE and THIRD_PARTY_NOTICES.md")
    for name in ("configs", "cluster", "tools", "huggingface", "artifacts"):
        if (root / name).exists():
            errors.append(f"Archived research surface remains: {name}")
    manifest_path = root / "src/emg_gpt/resources/artifacts.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("scope") != "inference_only" or manifest.get("schema_version") != 2:
            errors.append("Invalid inference artifact manifest")
        tokenizer = manifest["tokenizer"]
        source = tokenizer.get("public_source", {})
        if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
            errors.append("Tokenizer source is not pinned")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", tokenizer.get("sha256", ""))
            or tokenizer.get("bytes", 0) < 1
        ):
            errors.append("Tokenizer identity is missing")
        for task in ("regression", "tracking"):
            record = manifest["pose_models"][task]
            if not re.fullmatch(r"[0-9a-f]{64}", record.get("source_checkpoint_sha256", "")):
                errors.append(f"{task}: missing source checkpoint identity")
            source = record.get("source", {})
            if (
                source.get("provider") != "huggingface"
                or not source.get("repo_id")
                or source.get("subfolder") != task
                or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", ""))
            ):
                errors.append(f"{task}: missing or unpinned bundle source")
            if set(record.get("files", {})) != {
                "config.json",
                "model.safetensors",
                "codebooks.safetensors",
                "manifest.json",
            }:
                errors.append(f"{task}: incomplete bundle file manifest")
            for filename, identity in record.get("files", {}).items():
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", identity.get("sha256", ""))
                    or identity.get("bytes", 0) < 1
                ):
                    errors.append(f"{task}: missing hash or size for {filename}")
            if record.get("status") != "real_end_to_end_verified":
                errors.append(f"{task}: real end-to-end verification missing")
    return errors


def main() -> None:
    errors = check(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cffconvert.cli.cli import cli; cli()",
            "--validate",
            "-i",
            str(ROOT / "CITATION.cff"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        errors.append(
            "CITATION validation failed; install the dev extra and run cffconvert --validate"
        )
    print(json.dumps({"source_errors": errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
