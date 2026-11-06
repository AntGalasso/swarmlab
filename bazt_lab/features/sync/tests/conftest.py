"""Every test starts without a Hugging Face token and without a developer's real .env file."""
from __future__ import annotations

import pytest

from bazt_lab.features.sync.adapters import env


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "ENV_PATH", tmp_path / "absent.env")
    for name in ("HF_TOKEN", "BAZT_SYNC_FAKE_LOG", "BAZT_SYNC_FAKE_FAIL"):
        monkeypatch.setenv(name, "")             # recorded first, so a value a test loads from a .env file is undone
        monkeypatch.delenv(name)
