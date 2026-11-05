"""InjecAgent and PoisonedRAG: the files vendored in the G-Safeguard submodule, pinned by sha256."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.llm_data import public as llm_data
from bazt_lab.features.llm_data.private import vendored
from bazt_lab.features.llm_data.tests.conftest import injecagent_record, poisonedrag_record

TA = "vendor/G-safeguard/TA/datasets/attack_unsucc_data.json"
MA = "vendor/G-safeguard/MA/datasets/msmarco.json"


def draw_order(task, seed, n, part="all"):
    """The contract of the vendored draw: the order of n positions, from one pre-drawn stream keyed by strings."""
    return core.uniforms(core.stream("llm_data", "sample", task, part, seed), (n,)).argsort(stable=True)


def test_injecagent_items_carry_the_whole_record(settings, vendored_root):
    perm = draw_order("injecagent", 4, 10)[:3]
    got = llm_data.sample("injecagent", 4, 3, settings, root=vendored_root)
    assert [it.id for it in got] == [f"injecagent:{i}" for i in perm]
    it = got[0]
    assert (it.task, it.question, it.choices, it.answer) == (
        "injecagent", f"user instruction {perm[0]}", (), "")
    assert it.meta == {"record": injecagent_record(int(perm[0]))}


def test_poisonedrag_items_keep_native_ids_answers_and_poison(settings, vendored_root):
    perm = draw_order("poisonedrag_msmarco", 2, 10)[:4]
    got = llm_data.sample("poisonedrag_msmarco", 2, 4, settings, root=vendored_root)
    assert [it.id for it in got] == [f"poisonedrag_msmarco:{1000 + i}" for i in perm]
    rec = poisonedrag_record(int(perm[0]))
    it = got[0]
    assert (it.question, it.choices, it.answer) == (rec["question"], (), rec["correct answer"])
    assert it.meta == {"incorrect_answer": rec["incorrect answer"], "adv_texts": rec["adv_texts"],
                       "native_id": rec["id"]}


def test_source_records_the_vendored_path_and_sha256(settings, vendored_root):
    for task, path in (("injecagent", TA), ("poisonedrag_msmarco", MA)):
        src = llm_data.sample(task, 1, 2, settings, root=vendored_root)[0].source
        assert (src.dataset, src.config, src.split, src.revision) == (path, None, "all", None)
        assert src.sha256 == settings["llm_data"]["vendored_sha256"][task]


def test_sampling_is_deterministic_and_prefix_stable(settings, vendored_root):
    a = llm_data.sample("injecagent", 9, 5, settings, root=vendored_root)
    assert a == llm_data.sample("injecagent", 9, 5, settings, root=vendored_root)
    assert llm_data.sample("injecagent", 9, 8, settings, root=vendored_root)[:5] == a
    assert a != llm_data.sample("injecagent", 10, 5, settings, root=vendored_root)


def test_a_changed_file_is_refused_naming_both_digests(settings, vendored_root):
    path = vendored_root / TA
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(llm_data.DatasetMismatch) as err:
        llm_data.sample("injecagent", 1, 2, settings, root=vendored_root)
    msg = str(err.value)
    assert "llm_data.vendored_sha256.injecagent" in msg and settings["llm_data"]["vendored_sha256"]["injecagent"] in msg
    assert hashlib.sha256(path.read_bytes()).hexdigest() in msg


def test_git_autocrlf_line_endings_do_not_change_the_digest(settings, vendored_root):
    """A Windows checkout with core.autocrlf=true turns every LF into CRLF; the digest is taken over LF."""
    path = vendored_root / TA
    want = llm_data.sample("injecagent", 1, 3, settings, root=vendored_root)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert llm_data.sample("injecagent", 1, 3, settings, root=vendored_root) == want
    assert vendored.file_sha256(b"[1,\r\n2]\r\n") == hashlib.sha256(b"[1,\n2]\n").hexdigest()


def test_missing_digest_setting_names_the_key(settings, vendored_root):
    del settings["llm_data"]["vendored_sha256"]["poisonedrag_msmarco"]
    with pytest.raises(core.Unset) as err:
        llm_data.sample("poisonedrag_msmarco", 1, 2, settings, root=vendored_root)
    assert err.value.args[0] == "llm_data.vendored_sha256.poisonedrag_msmarco"


@pytest.mark.parametrize("digest", ["", "abc", "Z" * 64, "A" * 64, None])
def test_a_malformed_digest_setting_is_refused(digest, settings, vendored_root):
    settings["llm_data"]["vendored_sha256"]["injecagent"] = digest
    with pytest.raises(ValueError, match=r"llm_data.vendored_sha256.injecagent"):
        llm_data.sample("injecagent", 1, 2, settings, root=vendored_root)


def test_missing_file_says_how_to_fetch_the_submodule(settings, tmp_path):
    with pytest.raises(FileNotFoundError, match="submodule update --init"):
        llm_data.sample("injecagent", 1, 2, settings, root=tmp_path / "no-checkout")


def test_part_follows_the_upstream_80_20_positional_split(settings, vendored_root):
    train = llm_data.sample("injecagent", 1, 8, settings, part="train", root=vendored_root)
    test = llm_data.sample("injecagent", 1, 2, settings, part="test", root=vendored_root)
    assert sorted(int(it.id.split(":")[1]) for it in train) == list(range(8))        # int(10 * 0.8) = 8
    assert sorted(int(it.id.split(":")[1]) for it in test) == [8, 9]
    assert {it.source.split for it in train} == {"train"} and {it.source.split for it in test} == {"test"}
    with pytest.raises(ValueError):
        llm_data.sample("injecagent", 1, 3, settings, part="test", root=vendored_root)      # pool has 2
    with pytest.raises(ValueError, match="part"):
        llm_data.sample("injecagent", 1, 2, settings, part="dev", root=vendored_root)


def test_part_is_refused_for_hub_tasks(settings):
    with pytest.raises(ValueError, match="part"):
        llm_data.sample("mmlu", 1, 2, settings, part="train")


def test_k_larger_than_the_file_is_an_error(settings, vendored_root):
    with pytest.raises(ValueError, match="k"):
        llm_data.sample("poisonedrag_msmarco", 1, 11, settings, root=vendored_root)


def test_malformed_files_are_refused(settings, vendored_root):
    (vendored_root / TA).write_bytes(json.dumps({"not": "a list"}).encode())
    digest = hashlib.sha256((vendored_root / TA).read_bytes()).hexdigest()
    settings["llm_data"]["vendored_sha256"]["injecagent"] = digest
    with pytest.raises(ValueError, match="injecagent"):
        llm_data.sample("injecagent", 1, 1, settings, root=vendored_root)


def test_sources_lists_every_task_for_the_manifest(settings, vendored_root, ):
    got = llm_data.sources(settings, root=vendored_root)
    assert set(got) == set(llm_data.TASKS)
    assert got["mmlu"]["revision"] == settings["llm_data"]["revisions"]["mmlu"] and got["mmlu"]["sha256"] is None
    assert got["injecagent"]["sha256"] == settings["llm_data"]["vendored_sha256"]["injecagent"]
    assert got["poisonedrag_msmarco"]["dataset"] == MA
    json.dumps(got)


def test_vendored_draw_is_pinned_against_torch_drift(settings, vendored_root):
    assert draw_order("injecagent", 4, 10).tolist() == [7, 8, 3, 6, 5, 2, 0, 1, 4, 9]      # torch 2.14.0, CPU
    assert [it.id for it in llm_data.sample("injecagent", 4, 6, settings, root=vendored_root)] == [
        f"injecagent:{i}" for i in [7, 8, 3, 6, 5, 2]]


def test_vendored_draw_follows_the_named_stream_of_its_part(settings, vendored_root):
    ids = [int(it.id.split(":")[1]) for it in llm_data.sample("injecagent", 1, 8, settings, part="train",
                                                                 root=vendored_root)]
    assert ids == draw_order("injecagent", 1, 8, "train").tolist()
    assert ids != draw_order("injecagent", 1, 8).tolist()


def test_vendored_draw_never_touches_the_global_generators(settings, vendored_root):
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    llm_data.sample("injecagent", 4, 5, settings, root=vendored_root)
    llm_data.sample("poisonedrag_msmarco", 4, 5, settings, root=vendored_root)
    after = (torch.get_rng_state(), np.random.get_state()[1], random.getstate())
    assert torch.equal(before[0], after[0]) and np.array_equal(before[1], after[1]) and before[2] == after[2]


SHA256 = {"injecagent": "3465082b1d5bd32f60ba5e319a305f4b05bdb75dc00189e16ed6dc2937056998",
          "poisonedrag_msmarco": "d6bf508ebb0e31e09095995061bd483678e2ae8634f6ba5de1611e4adf9b1987"}
REPO = Path(__file__).resolve().parents[4]
needs_submodule = pytest.mark.skipif(not all((REPO / f).is_file() for f in (TA, MA)),
                                     reason="vendor/G-safeguard is not checked out (git submodule update --init)")


@needs_submodule
def test_the_real_submodule_files_are_the_pinned_ones_and_every_record_loads():
    real = {"llm_data": {"vendored_sha256": SHA256}}
    ta = llm_data.sample("injecagent", 1, 771, real)
    assert len(ta) == 771 and len({it.id for it in ta}) == 771
    assert {it.source.sha256 for it in ta} == {SHA256["injecagent"]}
    assert all(it.question == it.meta["record"]["User Instruction"] and it.meta["record"]["Attacker Tools"]
               for it in ta)
    ma = llm_data.sample("poisonedrag_msmarco", 1, 100, real)
    assert len({it.id for it in ma}) == 100 and {it.source.sha256 for it in ma} == {SHA256["poisonedrag_msmarco"]}
    assert all(it.id == f"poisonedrag_msmarco:{it.meta['native_id']}" and it.answer and it.meta["incorrect_answer"]
               and len(it.meta["adv_texts"]) == 5 for it in ma)
    with pytest.raises(ValueError, match="k"):
        llm_data.sample("injecagent", 1, 772, real)


@needs_submodule
def test_the_real_files_split_80_20_like_upstream():
    real = {"llm_data": {"vendored_sha256": SHA256}}
    assert len(llm_data.sample("injecagent", 1, 616, real, part="train")) == 616       # int(771 * 0.8)
    assert len(llm_data.sample("injecagent", 1, 155, real, part="test")) == 155
    assert len(llm_data.sample("poisonedrag_msmarco", 1, 80, real, part="train")) == 80
    assert len(llm_data.sample("poisonedrag_msmarco", 1, 20, real, part="test")) == 20
