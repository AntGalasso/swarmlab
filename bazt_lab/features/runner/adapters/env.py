"""Which secrets are present. Only presence is reported, never a value. Process environment first, then the repo-root
.env (python-dotenv, no override, nothing is loaded into the process)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

ENV_FILE = Path(__file__).resolve().parents[4] / ".env"
SECRETS = ("HF_TOKEN", "WANDB_API_KEY", "OPENROUTER_API_KEY")


def present(names: Iterable[str] = SECRETS) -> Dict[str, bool]:
    from dotenv import dotenv_values
    file = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    return {name: bool(os.environ.get(name) or file.get(name)) for name in names}
