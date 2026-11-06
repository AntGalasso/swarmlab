"""Seeded sampling of Hub tasks, with the Hub adapter replaced by a fake (offline)."""
import copy

import numpy as np
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.llm_data import public as llm_data
from bazt_lab.features.llm_data.adapters import hf
from bazt_lab.features.llm_data.tests.conftest import REV, ROWS

N_ROWS = 50
TASK_OF = {"cais/mmlu": "mmlu", "tau/commonsense_qa": "csqa", "openai/gsm8k": "gsm8k"}


@pytest.fixture
def calls(monkeypatch):
    """Replace hf.draw by a fake with the adapter's contract: k rows of the shuffled split, each with `_row`."""
    log = []

    def fake_draw(dataset, config, split, revision, seed, k):
        log.append((dataset, config, split, revision, seed, k))
        if k > N_ROWS:
            raise ValueError("k exceeds split")
        order = np.random.default_rng(seed).permutation(N_ROWS)[:k]
        return [{**ROWS[TASK_OF[dataset]](int(i)), "_row": int(i)} for i in order]

    monkeypatch.setattr(hf, "draw", fake_draw)
    return log


@pytest.mark.parametrize("task", ["mmlu", "csqa", "gsm8k"])
def test_same_seed_same_questions_other_seed_other_questions(task, settings, calls):
    a = llm_data.sample(task, 3, 12, settings)
    assert a == llm_data.sample(task, 3, 12, settings)
    assert a != llm_data.sample(task, 4, 12, settings)
    assert len(a) == 12 and len({it.id for it in a}) == 12 and {it.task for it in a} == {task}


def test_prefix_property_a_larger_k_extends_the_draw(settings, calls):
    short = llm_data.sample("mmlu", 3, 5, settings)
    assert llm_data.sample("mmlu", 3, 20, settings)[:5] == short


def test_adapter_gets_the_pinned_source_and_the_seed_itself(settings, calls):
    llm_data.sample("csqa", 7, 4, settings)
    assert calls == [("tau/commonsense_qa", "default", "validation", REV["csqa"], 7, 4)]
    llm_data.sample("mmlu", 7, 4, settings)
    assert calls[-1] == ("cais/mmlu", "all", "test", REV["mmlu"], 7, 4)
    llm_data.sample("gsm8k", 7, 4, settings)
    assert calls[-1] == ("openai/gsm8k", "main", "test", REV["gsm8k"], 7, 4)


def test_every_item_records_dataset_config_split_and_revision(settings, calls):
    for task, (dataset, config, split) in {"mmlu": ("cais/mmlu", "all", "test"),
                                           "csqa": ("tau/commonsense_qa", "default", "validation"),
                                           "gsm8k": ("openai/gsm8k", "main", "test")}.items():
        for it in llm_data.sample(task, 1, 3, settings):
            assert (it.source.dataset, it.source.config, it.source.split, it.source.revision) == (
                dataset, config, split, REV[task])


def test_item_order_is_the_draw_order(settings, calls):
    ids = [it.id for it in llm_data.sample("gsm8k", 5, 6, settings)]
    assert ids == [f"gsm8k:{i}" for i in np.random.default_rng(5).permutation(N_ROWS)[:6]]


def test_missing_revision_refuses_with_the_key_name(settings, calls):
    del settings["llm_data"]["revisions"]["gsm8k"]
    with pytest.raises(core.Unset) as err:
        llm_data.sample("gsm8k", 1, 3, settings)
    assert err.value.args[0] == "llm_data.revisions.gsm8k" and calls == []


@pytest.mark.parametrize("rev", ["main", "c30699e", "C30699E8356DA336A370243923DBAF21066BB9FE", "", None, 7,
                                 REV["mmlu"] + "0"])
def test_a_revision_that_is_not_a_full_commit_sha_is_refused(rev, settings, calls):
    settings["llm_data"]["revisions"]["mmlu"] = rev
    with pytest.raises(ValueError, match=r"llm_data.revisions.mmlu"):
        llm_data.sample("mmlu", 1, 3, settings)
    assert calls == []


@pytest.mark.parametrize("seed, k", [(-1, 3), (1, 0), (1, -2), (1.5, 3), (1, 2.0), (True, 3), ("1", 3)])
def test_bad_seed_or_k_is_refused_before_any_load(seed, k, settings, calls):
    with pytest.raises(ValueError):
        llm_data.sample("mmlu", seed, k, settings)
    assert calls == []


def test_k_larger_than_the_split_is_an_error(settings, calls):
    with pytest.raises(ValueError):
        llm_data.sample("mmlu", 1, N_ROWS + 1, settings)


def test_unknown_task_lists_the_known_ones(settings, calls):
    with pytest.raises(ValueError, match=r"mmlu.*csqa.*gsm8k"):
        llm_data.sample("squad", 1, 3, settings)


def test_settings_are_not_mutated(settings, calls):
    before = copy.deepcopy(settings)
    llm_data.sample("mmlu", 1, 3, settings)
    assert settings == before


def test_task_names_match_the_attack_cells_of_the_profile():
    cells = core.get(core.load_settings(), "llm.attack_cells")
    assert {task for _, task in cells} <= set(llm_data.TASKS)


def test_profiles_toml_carries_the_requested_pins():
    """Once the coordinator applies the CONTRACT.md Requests, the pins in profiles.toml must be exactly these."""
    from bazt_lab.features.llm_data.tests.test_vendored import SHA256

    live = core.load_settings()
    if not core.is_set(live, "llm_data"):
        pytest.skip("experiments/profiles.toml has no [llm_data] table yet (CONTRACT.md, Requests)")
    assert core.get(live, "llm_data.revisions") == REV
    assert core.get(live, "llm_data.vendored_sha256") == SHA256
