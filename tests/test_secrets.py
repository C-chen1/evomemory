"""Guard against secrets leaking into tracked files.

This exists because it already happened once during development: the API key was
pasted into ``.env.example`` — a *tracked* template — instead of ``.env``. The
mistake was caught before it reached a commit, but only because someone looked.
A test does not get tired.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Key-shaped strings that must never appear in a tracked file.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b[A-Za-z0-9_\-]*api[_-]?key\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}",
               re.IGNORECASE),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
]

TRACKED_SUFFIXES = {".py", ".md", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".example"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "references",
             "artifacts", "data", "cache", "models", "logs", "checkpoints"}


def _walk(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in TRACKED_SUFFIXES or path.name.startswith(".env"):
            yield path


def test_no_secret_shaped_strings_in_project_files():
    offenders = []
    for path in _walk(ROOT):
        if path.name == ".env":           # git-ignored, allowed to hold the key
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                # allow obvious documentation placeholders
                if any(tok in snippet.lower() for tok in
                       ("placeholder", "your_key", "xxxx", "<key>", "example")):
                    continue
                offenders.append(f"{path.relative_to(ROOT)}: {snippet[:8]}…")
    assert not offenders, "possible secrets in tracked files:\n" + "\n".join(offenders)


def test_env_example_has_empty_values():
    example = ROOT / ".env.example"
    assert example.exists(), ".env.example must ship as a template"
    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() in {"DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"}:
            assert value.strip() == "", (
                f"{name} must be empty in .env.example; real values belong in .env")


def test_env_is_gitignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    entries = [line.strip() for line in gitignore
               if line.strip() and not line.startswith("#")]
    assert ".env" in entries
    assert any(entry.startswith("!.env.example") for entry in entries)
