"""Check distributable files for broken local links and accidental private content."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "local path": r"/(?:Users|home|scratch)/[A-Za-z0-9_.-]+/",
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b",
}


def check(root: Path) -> list[str]:
    if (root / ".git").exists():
        names = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ]
            )
            .decode()
            .split("\0")
        )
        paths = sorted({root / name for name in names if name})
    else:
        # A source archive has no Git index; inspect only its distributable directories.
        paths = [p for p in root.iterdir() if p.is_file()]
        for directory in ("src", "tests", "scripts", "docs", "constraints", "licenses", "images"):
            paths.extend((root / directory).rglob("*"))
    errors = []
    for path in paths:
        if not path.is_file() or any(part.endswith(".egg-info") for part in path.parts):
            continue
        name = path.relative_to(root).as_posix()
        if path.suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5"}:
            errors.append(f"Weight or dataset included in source: {name}")
        if path.suffix not in {".py", ".md", ".json", ".toml", ".yml", ".cff"}:
            continue
        text = path.read_text()
        for label, pattern in PATTERNS.items():
            if re.search(pattern, text):
                errors.append(f"Possible {label}: {name}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" not in target and not target.startswith("#"):
                    if not (path.parent / target.split("#")[0]).exists():
                        errors.append(f"Broken local link: {name} -> {target}")
    return errors


if __name__ == "__main__":
    errors = check(ROOT)
    print("\n".join(errors) if errors else "Source checks passed")
    raise SystemExit(bool(errors))
