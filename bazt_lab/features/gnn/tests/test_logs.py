"""Conversation logs: the files the LLM platform writes and the trainer reads, one JSONL line per conversation."""
import json

import pytest

from bazt_lab.features.gnn.private.logs import log_conversations, read_conversations
from bazt_lab.features.gnn.tests.conftest import make_records


def unit(uid="GNN-aaa", attack="insider_pi", task="mmlu"):
    return {"block": "gnn_train", "platform": "llm", "arm": "GNN", "attack": attack, "task": task, "id": uid,
            "edge_probability": 0.6, "attackers": 1, "backbone": "test-backbone", "conversations": 6, "agents": 6,
            "rounds": 4}


def test_a_logged_conversation_is_read_back_unchanged(settings):
    records = make_records(3, seed=1)
    log_conversations(settings, unit(), records)
    assert read_conversations(settings, "insider_pi__mmlu") == records


def test_several_units_of_a_family_are_read_together_in_file_name_order(settings):
    a, b = make_records(2, seed=1), make_records(2, seed=2)
    log_conversations(settings, unit("GNN-bbb"), b)
    log_conversations(settings, unit("GNN-aaa"), a)
    assert read_conversations(settings, "insider_pi__mmlu") == a + b


def test_a_family_with_no_log_reads_as_empty_and_creates_nothing(settings, tmp_path):
    assert read_conversations(settings, "insider_ta__injecagent") == []
    assert not (tmp_path / "logs").exists()


def test_other_families_are_not_mixed_in(settings):
    log_conversations(settings, unit("GNN-aaa"), make_records(2, seed=1))
    log_conversations(settings, unit("GNN-ccc", "insider_ta", "injecagent"),
                      make_records(2, seed=3, attack="insider_ta", task="injecagent"))
    assert [r["id"] for r in read_conversations(settings, "insider_ta__injecagent")] == ["s3-c0", "s3-c1"]


def test_a_repeated_conversation_is_kept_once(settings):
    """A launch resumed after a crash may log a conversation twice; it is the same conversation (same id)."""
    records = make_records(3, seed=1)
    log_conversations(settings, unit(), records)
    log_conversations(settings, unit(), records[1:])
    assert read_conversations(settings, "insider_pi__mmlu") == records


def test_a_torn_last_line_is_dropped_on_read(settings, tmp_path):
    records = make_records(2, seed=1)
    log_conversations(settings, unit(), records)
    path = tmp_path / "logs" / "insider_pi__mmlu" / "units" / "GNN-aaa.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "torn", "adj"')
    assert read_conversations(settings, "insider_pi__mmlu") == records


def test_a_malformed_record_is_refused_with_its_id_and_nothing_is_written(settings, tmp_path):
    good, bad = make_records(2, seed=1)
    bad["attackers"] = [99]
    with pytest.raises(ValueError, match=rf"{bad['id']}.*attacker"):
        log_conversations(settings, unit(), [good, bad])
    assert read_conversations(settings, "insider_pi__mmlu") == []


def test_a_record_of_another_family_than_the_unit_is_refused(settings):
    (rec,) = make_records(1, seed=1, attack="insider_ta", task="injecagent")
    with pytest.raises(ValueError, match=rf"{rec['id']}.*insider_ta__injecagent.*insider_pi__mmlu"):
        log_conversations(settings, unit(), [rec])


def test_a_record_must_name_its_backbone_and_family(settings):
    (rec,) = make_records(1, seed=1)
    del rec["backbone"]
    with pytest.raises(ValueError, match=rf"{rec['id']}.*backbone"):
        log_conversations(settings, unit(), [rec])


def test_the_log_is_plain_json_lines(settings, tmp_path):
    records = make_records(2, seed=1)
    log_conversations(settings, unit(), records)
    lines = (tmp_path / "logs" / "insider_pi__mmlu" / "units" / "GNN-aaa.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records


def test_a_missing_logs_dir_setting_is_named(settings):
    from bazt_lab.core import public as core
    del settings["gnn_train"]["logs_dir"]
    with pytest.raises(core.Unset, match=r"gnn_train.logs_dir"):
        read_conversations(settings, "insider_pi__mmlu")
