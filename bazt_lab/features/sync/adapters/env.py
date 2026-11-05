"""The process environment: the only place in this feature that touches HF_TOKEN and the .env file. The token is never
read from settings and never printed: errors name the variable, `redact` removes the value from text about to be shown."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from bazt_lab.features.sync.private.errors import SyncError

ENV_PATH = Path(__file__).resolve().parents[4] / ".env"
VARIABLE = "HF_TOKEN"


def load() -> None:
    load_dotenv(ENV_PATH)                    # a missing file is fine; the process environment wins (no override)


def require_token() -> None:
    load()
    if not os.environ.get(VARIABLE):
        raise SyncError(f"sync is enabled and {VARIABLE} is not set. Export it or put it in the repo-root .env (see "
                        ".env.example); it is never read from experiments/profiles.toml.")


def redact(text: str) -> str:
    token = os.environ.get(VARIABLE)
    return text.replace(token, "[redacted]") if token else text
