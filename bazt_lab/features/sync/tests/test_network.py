"""One real round trip to the Hub. Skipped unless BAZT_NETWORK_TESTS=1, with the Hub token in the environment or the
repo-root .env and BAZT_SYNC_TEST_REPO (a private scratch dataset repo you own). Never runs by default."""
from __future__ import annotations

import os

import pytest

from bazt_lab.features.sync import public as sync
from bazt_lab.features.sync.adapters import env, hub


@pytest.mark.skipif(os.environ.get("BAZT_NETWORK_TESTS") != "1", reason="set BAZT_NETWORK_TESTS=1 to reach the Hub")
def test_a_real_push_lands_in_the_private_scratch_repo(tmp_path, monkeypatch):
    monkeypatch.undo()                                   # the autouse fixture hid the real token and .env
    env.load()
    repo = os.environ["BAZT_SYNC_TEST_REPO"]
    folder = tmp_path / "ckpt"
    folder.mkdir()
    (folder / "state.json").write_text('{"ok": true}', encoding="utf-8")
    (folder / "state.json.tmp").write_text("partial", encoding="utf-8")
    keys = {"enabled": True, "repo_id": repo, "repo_type": "dataset", "private": True, "every_minutes": 60,
            "include": [], "exclude": [], "priority": "idle", "final_push_timeout_seconds": 300}
    handle = sync.start({"sync": keys}, [folder])
    handle.stop()
    files = hub.make_uploader({"settings": keys}).api.list_repo_files(repo, repo_type="dataset")
    assert "ckpt/state.json" in files and "ckpt/state.json.tmp" not in files
    assert handle.summary["failures"] == {}
