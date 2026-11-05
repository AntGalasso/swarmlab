"""The patch set applies cleanly to the pinned G-Safeguard checkout, on a derived copy, never to the submodule."""
import shutil
import subprocess

import pytest

from bazt_lab.features.llm_harness.adapters import upstream

REPO = upstream.REPO


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return upstream.prepare(tmp_path_factory.mktemp("gs") / "patched")


def test_submodule_is_pinned_and_untouched_by_prepare(prepared):
    head = subprocess.run(["git", "-C", str(upstream.SUBMODULE), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == upstream.PINNED_COMMIT == "890c99f1cbc864e9ff0c85859619a14f42bc9cab"
    dirty = subprocess.run(["git", "-C", str(upstream.SUBMODULE), "status", "--porcelain", "--", "PI", "TA", "MA",
                            "scalability"], capture_output=True, text=True, check=True).stdout
    assert dirty.strip() == ""


def test_every_patch_lands_in_the_derived_copy(prepared):
    pi = (prepared / "PI" / "agents.py").read_text(encoding="utf-8")
    assert "def delivered_view" in pi and "self.outbox" in pi and "self.context" in pi          # 0004, 0005, 0006
    main = (prepared / "PI" / "main_defense_for_different_topology.py").read_text(encoding="utf-8")
    assert "hook" in main and "MyGAT" not in main and "scatter_mean" not in main                # 0003
    assert '"--seed"' in main                                                                   # 0001
    ta_utils = (prepared / "TA" / "utils.py").read_text(encoding="utf-8")
    assert "nltk" not in ta_utils and "__file__" in ta_utils                                    # 0002
    assert (prepared / "TA" / "tools.json").is_file()


def test_derived_copy_has_unix_line_endings_whatever_the_checkout_uses(prepared):
    for path in prepared.rglob("*.py"):
        assert b"\r" not in path.read_bytes(), path


def test_prepare_refuses_a_checkout_at_another_commit(tmp_path):
    with pytest.raises(upstream.HarnessError, match="890c99f"):
        upstream.prepare(tmp_path / "x", expect_commit="0" * 40)


def test_a_patch_that_does_not_apply_stops_before_anything_is_written(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(upstream.SUBMODULE, src, ignore=shutil.ignore_patterns("datasets", "data", "assets", ".git",
                                                                          "__pycache__", "scripts"))
    victim = src / "PI" / "agents.py"
    victim.write_bytes(victim.read_bytes().replace(b"self.attacker_idxes = attacker_idxes",
                                                   b"self.attackers = attacker_idxes"))
    with pytest.raises(upstream.HarnessError, match=r"0004-delivery-masks\.patch"):
        upstream.prepare(tmp_path / "out", source=src, expect_commit=None)
    assert not (tmp_path / "out").exists()


def test_prepare_twice_reuses_the_copy(tmp_path):
    first = upstream.prepare(tmp_path / "p")
    stamp = (first / ".harness").read_text(encoding="utf-8")
    marker = first / "PI" / "keep.txt"
    marker.write_text("x")
    again = upstream.prepare(tmp_path / "p")
    assert again == first and marker.exists() and (again / ".harness").read_text(encoding="utf-8") == stamp


def test_a_stale_copy_is_rebuilt(tmp_path):
    first = upstream.prepare(tmp_path / "p")
    (first / ".harness").write_text("other", encoding="utf-8")
    marker = first / "PI" / "keep.txt"
    marker.write_text("x")
    upstream.prepare(tmp_path / "p")
    assert not marker.exists()


def test_patch_list_matches_the_files_on_disk():
    for patch in upstream.PATCHES:
        path = upstream.PATCH_DIR / patch
        assert path.is_file(), path
    assert [p.name for p in sorted(upstream.PATCH_DIR.glob("*.patch"))] == list(upstream.PATCHES)


def test_prepare_applies_the_patches_inside_a_git_repository_that_ignores_the_destination(tmp_path):
    """results/experiments/ is gitignored in this repository; git apply run there silently skipped every file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("out/\n", encoding="utf-8")
    patched = upstream.prepare(repo / "out" / "patched")
    main = (patched / "PI" / "main_defense_for_different_topology.py").read_text(encoding="utf-8")
    assert "torch_scatter" not in main and "hook" in main
