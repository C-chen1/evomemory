"""Configuration, loaded from the project ``.env`` and the environment.

Rules taken from the brief and enforced here:

* no key is ever hardcoded, logged, or written into any artifact;
* the model id has exactly one entry point (``Settings.model``);
* the API budget has a hard cap that is checked *before* a call is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path, override: bool = False) -> Dict[str, str]:
    """Minimal ``.env`` parser. ``KEY=value`` and ``# comment`` only."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


@dataclass
class Settings:
    model: str = ""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    max_calls: int = 200
    max_tokens: int = 500_000
    dry_run: bool = True
    root: Path = field(default_factory=lambda: PROJECT_ROOT)

    # -- derived paths ---------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "evomemory.db"

    @property
    def evidence_dir(self) -> Path:
        return self.data_dir / "evidence"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    # -- construction ----------------------------------------------------
    @classmethod
    def from_env(cls, root: Optional[Path] = None, dotenv: Optional[Path] = None) -> "Settings":
        root = Path(root) if root else PROJECT_ROOT
        load_dotenv(dotenv or (root / ".env"))
        api_key = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("DEEPSEEK_MODEL", "")
        base_url = (os.getenv("DEEPSEEK_BASE_URL")
                    or os.getenv("OPENAI_BASE_URL")
                    or "https://api.deepseek.com")
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            temperature=float(os.getenv("EVOMEMORY_TEMPERATURE", "0.0")),
            max_calls=int(os.getenv("EVOMEMORY_MAX_CALLS", "200")),
            max_tokens=int(os.getenv("EVOMEMORY_MAX_TOKENS", "500000")),
            dry_run=not bool(api_key and model),
            root=root,
        )

    # -- safety ----------------------------------------------------------
    def describe(self) -> Dict[str, object]:
        """A description safe to print, log or commit."""
        return {
            "model": self.model or None,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "api_key_fingerprint": (f"…{self.api_key[-4:]}" if self.api_key else None),
            "temperature": self.temperature,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "dry_run": self.dry_run,
            "db_path": str(self.db_path),
        }

    def require_api(self) -> None:
        if self.dry_run:
            raise RuntimeError(
                "No DEEPSEEK_API_KEY / DEEPSEEK_MODEL configured. "
                "Write them into .env at the project root, or run with the offline "
                "scripted client. Refusing to guess a model id."
            )
