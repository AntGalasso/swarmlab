"""Which files go to the Hub: the mirror layout, the exclusions and the change detection."""
from __future__ import annotations

import os

from bazt_lab.features.sync.private import plan


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def names(entries):
    return sorted(e.repo_path for e in entries)


def test_each_path_is_mirrored_under_a_folder_named_after_it(tmp_path):
    write(tmp_path / "results" / "mock_main" / "rows.jsonl")
    write(tmp_path / "results" / "mock_main" / "sub" / "a.json")
    write(tmp_path / "gnn_ckpt" / "epoch1.pt")
    got = plan.snapshot([tmp_path / "results", tmp_path / "gnn_ckpt"], include=[], exclude=[])
    assert names(got) == ["gnn_ckpt/epoch1.pt", "results/mock_main/rows.jsonl", "results/mock_main/sub/a.json"]


def test_a_single_file_is_mirrored_under_its_own_name(tmp_path):
    write(tmp_path / "thresholds.json")
    assert names(plan.snapshot([tmp_path / "thresholds.json"], [], [])) == ["thresholds.json/thresholds.json"]


def test_tmp_files_are_never_uploaded_even_when_include_names_them(tmp_path):
    write(tmp_path / "r" / "state.json")
    write(tmp_path / "r" / "state.json.tmp")
    write(tmp_path / "r" / "sub" / "x.tmp")
    assert names(plan.snapshot([tmp_path / "r"], include=["*"], exclude=[])) == ["r/state.json"]


def test_exclude_patterns_match_the_file_name_or_the_path_under_the_folder(tmp_path):
    write(tmp_path / "r" / ".launch.lock")
    write(tmp_path / "r" / "cache" / "big.bin")
    write(tmp_path / "r" / "keep.json")
    got = plan.snapshot([tmp_path / "r"], include=[], exclude=[".launch.lock", "cache/*"])
    assert names(got) == ["r/keep.json"]


def test_include_patterns_limit_the_upload_when_given(tmp_path):
    write(tmp_path / "r" / "rows.jsonl")
    write(tmp_path / "r" / "notes.txt")
    assert names(plan.snapshot([tmp_path / "r"], include=["*.jsonl"], exclude=[])) == ["r/rows.jsonl"]


def test_a_path_that_does_not_exist_yet_is_skipped_not_an_error(tmp_path):
    assert plan.snapshot([tmp_path / "later"], [], []) == []


def test_only_new_or_changed_files_are_selected_once_marked_uploaded(tmp_path):
    a, b = write(tmp_path / "r" / "a.json", "1"), write(tmp_path / "r" / "b.json", "1")
    entries = plan.snapshot([tmp_path / "r"], [], [])
    uploaded = {}
    assert names(plan.changed(entries, uploaded)) == ["r/a.json", "r/b.json"]
    uploaded.update(plan.stamps(entries))
    assert plan.changed(plan.snapshot([tmp_path / "r"], [], []), uploaded) == []
    a.write_text("2222", encoding="utf-8")
    os.utime(a, ns=(a.stat().st_atime_ns, a.stat().st_mtime_ns + 5_000_000_000))
    write(tmp_path / "r" / "c.json")
    assert names(plan.changed(plan.snapshot([tmp_path / "r"], [], []), uploaded)) == ["r/a.json", "r/c.json"]
    assert b.exists()


def test_a_file_that_grows_with_the_same_mtime_counts_as_changed(tmp_path):
    f = write(tmp_path / "r" / "rows.jsonl", "a\n")
    before = plan.snapshot([tmp_path / "r"], [], [])
    uploaded = plan.stamps(before)
    mtime = f.stat().st_mtime_ns
    f.write_text("a\nb\n", encoding="utf-8")
    os.utime(f, ns=(mtime, mtime))
    assert names(plan.changed(plan.snapshot([tmp_path / "r"], [], []), uploaded)) == ["r/rows.jsonl"]
