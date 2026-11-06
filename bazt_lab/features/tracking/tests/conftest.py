"""Every test starts without a W&B key, without a kickoff override and without a developer's real .env file."""
from __future__ import annotations

import pytest

from bazt_lab.features.tracking.adapters import env


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "ENV_PATH", tmp_path / "absent.env")
    for name in ("WANDB_API_KEY", "SWARMLAB_KICKOFF_UTC"):
        monkeypatch.setenv(name, "")         # recorded first, so a value a test loads from a .env file is undone
        monkeypatch.delenv(name)
