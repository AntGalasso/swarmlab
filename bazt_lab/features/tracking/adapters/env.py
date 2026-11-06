"""The process environment: the only place in this feature that touches WANDB_API_KEY and the .env file.

The key is never read from settings and never printed: errors name the variable, and `redact` removes the value from
any text that is about to be shown. A repo-root `.env` fills what the process leaves unset (python-dotenv never
overrides a variable the process already has).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from bazt_lab.features.tracking.private.errors import TrackingError

ENV_PATH = Path(__file__).resolve().parents[4] / ".env"
API_KEY = "WANDB_API_KEY"


def load() -> None:
    load_dotenv(ENV_PATH)                   # a missing file is fine


def require_api_key() -> None:
    load()
    if not os.environ.get(API_KEY):
        raise TrackingError(f"W&B runs online and {API_KEY} is not set. Export it or put it in the repo-root .env "
                            "(see .env.example); it is never read from experiments/profiles.toml.")


def redact(text: str) -> str:
    key = os.environ.get(API_KEY)
    return text.replace(key, "[redacted]") if key else text
