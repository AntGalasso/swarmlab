"""Second fix round: fingerprint vs the runner's block identity, torn progress tails, platform row validation, settings
validation, stale thresholds refused before any search, plan memory."""
import gc
import json
import weakref

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.adaptive import public as adaptive
from bazt_lab.features.adaptive.private import run as rn
from bazt_lab.features.adaptive.private import search as se
from bazt_lab.features.adaptive.private import space as sp
from bazt_lab.features.adaptive.tests.fakes import FakePlatform, arm_of, drop, make_settings, unit, units_of

N = 6


# ------------------------------------------------------------------------------------------------ R2-M1
def test_the_fingerprint_ignores_what_the_runners_block_identity_ignores():
    base = rn.fingerprint(make_settings())
    for over in ({"sync.enabled": True}, {"sync.every_minutes": 5}, {"tracking.mode": "online"}, {"tracking.enabled": True},
                 {"runtime.random_edge_probability": 0.3}):
        assert rn.fingerprint(make_settings(over)) == base, over


# ------------------------------------------------------------------------------------------------ torn progress
def _job():
    cell = dict(arm="X", attack="infection", agents=N, condition="gray_box")
    pairs = tuple((s, None) for s in (1, 2, 3))
    return se.Job(cell, "cell", pairs, {(None, None): (0, 1, 2)}, {p: f"id{p[0]}" for p in pairs})


CANDS = [{}, {"entry_ranks": [0]}, {"entry_ranks": [1]}]


def _asr(u):
    return (len(u["id"]) + len(str(u["attack_params"]))) % 7 / 6.0


def _run(limit):
    calls = []

    def run(batch):
        if len(calls) >= limit:
            raise KeyboardInterrupt
        calls.append(len(batch))
        return [_asr(u) for u in batch]
    return run, calls


def test_a_valid_progress_line_without_its_newline_is_dropped_so_the_next_append_is_not_glued_to_it(tmp_path):
    job, prog = _job(), tmp_path / "p.jsonl"
    full = se.evaluate(job, CANDS, _run(10**9)[0], tmp_path / "full.jsonl", 2, lambda n: None, {})
    with pytest.raises(KeyboardInterrupt):
        se.evaluate(job, CANDS, _run(2)[0], prog, 2, lambda n: None, {})
    prog.write_text(prog.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")     # crash before the newline
    with pytest.raises(KeyboardInterrupt):
        se.evaluate(job, CANDS, _run(1)[0], prog, 2, lambda n: None, {})
    text = prog.read_text(encoding="utf-8")
    assert text.endswith("\n") and all(json.loads(line) for line in text.splitlines())
    run, calls = _run(10**9)
    assert torch.equal(se.evaluate(job, CANDS, run, prog, 2, lambda n: None, {}), full)
    assert sum(calls) <= len(CANDS) * 3


# ------------------------------------------------------------------------------------------------ platform rows
@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), True, -0.1, 1.5])
def test_an_asr_that_is_not_a_finite_number_in_0_1_is_refused(tmp_path, bad):
    class Raw(FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            rows = super().run_batch(units, arm, settings, store, tracker)
            return [{**r, "asr_final": bad} for r in rows]
    with pytest.raises(ValueError, match="asr_final"):
        rn.run_search(units_of(unit(agents=N), (1,), (0,)), arm_of, Raw(lambda arm, u: 0.5), make_settings(), tmp_path)


# ------------------------------------------------------------------------------------------------ settings
@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_n_insiders_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="n_insiders"):
        sp.load_space(make_settings({"runtime.n_insiders": bad}))


@pytest.mark.parametrize("key", ["start_fractions", "rates", "detectabilities"])
def test_nan_in_a_grid_is_refused(key):
    with pytest.raises(ValueError, match=key):
        sp.load_space(make_settings({f"mock.adaptive.{key}": [1.0, float("nan")]}))


def test_search_seeds_are_required_only_when_a_cell_level_condition_is_configured():
    s = make_settings({"mock.adaptive.conditions": ["white_box"]})
    assert sp.load_space(drop(s, "mock.adaptive.search_seeds")).search_seeds is None
    with pytest.raises(core.Unset) as e:
        sp.load_space(drop(make_settings(), "mock.adaptive.search_seeds"))
    assert e.value.args[0] == "mock.adaptive.search_seeds"


def test_a_unit_without_a_seed_is_refused_by_name():
    u = units_of(unit(agents=N), (1,), (0,))
    del u[0]["seed"]
    with pytest.raises(sp.Refusal, match="seed"):
        se.plan_jobs(u, sp.load_space(make_settings()))


# ------------------------------------------------------------------------------------------------ preflight
def test_stale_thresholds_of_any_cell_are_refused_before_anything_is_searched(tmp_path):
    settings = make_settings({"mock.batch_size": 100})
    old = units_of(unit(agents=N, audit=0.2), (1,), (0,))
    rn.run_search(old, arm_of, FakePlatform(lambda arm, u: 0.5), settings, tmp_path, thresholds_of=lambda c: {"m": 1.0})
    fresh_first = units_of(unit(agents=N, audit=0.05), (1,), (0,))
    platform = FakePlatform(lambda arm, u: 0.5)
    with pytest.raises(ValueError, match="other thresholds"):
        rn.run_search(fresh_first + old, arm_of, platform, settings, tmp_path, thresholds_of=lambda c: {"m": 1.1})
    assert platform.calls == [] and not list((tmp_path / rn.PROGRESS_DIR).glob("*.jsonl"))


# ------------------------------------------------------------------------------------------------ plan memory
class Tracked(dict):
    pass


def test_planning_does_not_keep_the_evaluation_units_alive():
    refs = []

    def stream():
        for u in units_of(unit(agents=N, condition="white_box"), range(1, 6), range(N)):
            t = Tracked(u)
            refs.append(weakref.ref(t))
            yield t
    jobs = se.plan_jobs(stream(), sp.load_space(make_settings()))
    gc.collect()
    assert len(refs) == 5 * N and all(r() is None for r in refs) and len(jobs[0].pairs) == 5 * N
    assert adaptive.search_episodes(stream(), make_settings()) > 0
