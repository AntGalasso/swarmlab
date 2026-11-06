"""Fix round: common random numbers per episode (same id for every candidate), exact integer ties, empty resume,
extra seed and fingerprint rules, refusals the runner prints, quiet tracker."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.adaptive import public as adaptive
from bazt_lab.features.adaptive.private import run as rn
from bazt_lab.features.adaptive.private import search as se
from bazt_lab.features.adaptive.private import table as tb
from bazt_lab.features.adaptive.tests.fakes import FakePlatform, arm_of, make_settings, unit, units_of
from bazt_lab.features.adaptive.tests.test_mock_platform import engine_settings
from bazt_lab.features.mock_engine import public as mock_engine

N = 6


# ------------------------------------------------------------------------------------------------ M1
def ids_by_pair(platform):
    seen = {}
    for call in platform.calls:
        assert len({u["id"] for u in call["units"]}) == len(call["units"]), "duplicate id inside one batch"
        for u in call["units"]:
            seen.setdefault((u["seed"], u["sink"]), set()).add(u["id"])
    return seen


def test_every_candidate_of_one_search_episode_has_the_same_id_and_episodes_have_different_ids(tmp_path):
    platform = FakePlatform(lambda arm, u: 0.5)
    units = units_of(unit(agents=N), (1, 2), range(N))
    adaptive.run_search(units, arm_of, platform, make_settings({"mock.batch_size": 100}), tmp_path)
    seen = ids_by_pair(platform)
    assert all(len(v) == 1 for v in seen.values()) and len({next(iter(v)) for v in seen.values()}) == len(seen)


def test_a_white_box_search_runs_on_the_ids_of_the_evaluation_units(tmp_path):
    platform = FakePlatform(lambda arm, u: 0.5)
    units = [{**u, "id": f"eval-{u['seed']}-{u['sink']}"}
             for u in units_of(unit(agents=N, condition="white_box"), (1, 2), (0, 3))]
    adaptive.run_search(units, arm_of, platform, make_settings({"mock.batch_size": 100}), tmp_path)
    assert {k: next(iter(v)) for k, v in ids_by_pair(platform).items()} == {(u["seed"], u["sink"]): u["id"] for u in units}


class RandomIsolate:
    """Draws from the arm stream: isolates one agent chosen by the episode's own arm-keyed generator at round 1."""
    name = "random_isolate"

    def reset(self, ctx):
        self.ctx = ctx

    def decide(self, obs):
        iso = torch.zeros_like(obs.active)
        if obs.t == 1:
            for b in range(self.ctx.B):
                pick = int(core.uniforms(self.ctx.arm_stream("pick", b), (1,))[0] * self.ctx.N)
                iso[b, pick] = True
        return core.Actions(isolation=iso & obs.active)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


RARM = core.ArmSpec(name="random_isolate", factory=RandomIsolate)


def test_the_searched_asr_of_an_arm_with_arm_keyed_randomness_equals_its_evaluated_asr(tmp_path):
    settings = engine_settings()
    units = [dict(block="mock_adaptive", platform="mock", arm=RARM.name, audit=0.1, intervention="uncapped",
                  topology="small_world", attack="infection", agents=N, seed=s, sink=k, condition="white_box")
             for s in (1, 2, 3) for k in (0, 3)]
    units = [{**u, "id": f"{u['seed']}-{u['sink']}-eval"} for u in units]
    table = adaptive.run_search(units, lambda n: RARM, mock_engine.MockPlatform(), settings, tmp_path)
    rows = mock_engine.MockPlatform().run_batch([{**table.annotate(u), "thresholds": {}} for u in units], RARM, settings,
                                                core.CheckpointStore(tmp_path / "eval"), core.NullTracker())
    for u, r in zip(units, rows, strict=True):
        assert r["asr_final"] == table.choice(u).asr


# ------------------------------------------------------------------------------------------------ M2
def test_equal_integer_totals_of_wrong_agents_tie_and_the_earlier_candidate_wins():
    n = 10
    cell = dict(arm="X", attack="infection", agents=n, condition="gray_box")
    job = se.Job(cell, "cell", ((1, None), (2, None), (3, None)), {(None, None): (0, 1, 2)}, {})
    vals = torch.tensor([[3 / n, 0.0, 0.0], [1 / n, 1 / n, 1 / n]], dtype=torch.float64)
    # [0.3, 0, 0] vs [0.1, 0.1, 0.1]: both 3 wrong agents; the float means differ by one ulp
    assert se.choose(job, [{}, {"entry_ranks": [0]}], vals, n_victims=n)[0]["attack_params"] == {}
    vals = torch.tensor([[1 / n, 1 / n, 1 / n], [3 / n, 0.0, 0.0]], dtype=torch.float64)
    assert se.choose(job, [{}, {"entry_ranks": [0]}], vals, n_victims=n)[0]["attack_params"] == {}


# ------------------------------------------------------------------------------------------------ M3
def test_a_complete_progress_file_without_a_table_file_is_finished_without_simulating(tmp_path, monkeypatch):
    units = units_of(unit(agents=N), (1, 2), range(N))
    settings = make_settings({"mock.batch_size": 40})
    real = tb.write_cell

    def crash(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(tb, "write_cell", crash)
    with pytest.raises(KeyboardInterrupt):
        rn.run_search(units, arm_of, FakePlatform(lambda arm, u: 0.5), settings, tmp_path)
    monkeypatch.setattr(tb, "write_cell", real)
    platform = FakePlatform(lambda arm, u: 0.5)
    table = rn.run_search(units, arm_of, platform, settings, tmp_path)
    assert platform.calls == [] and len(table) == 1


# ------------------------------------------------------------------------------------------------ minors
def test_search_seeds_must_not_overlap_the_tune_seeds():
    with pytest.raises(ValueError, match="tune_seeds"):
        adaptive.load_space(make_settings({"mock.adaptive.search_seeds": [9020, 9040]}))


def test_the_fingerprint_ignores_unrelated_subtrees():
    base = rn.fingerprint(make_settings())
    for over in ({"analysis.alpha": 0.1}, {"gnn.hidden": 8}, {"llm_data.rev": "a"}):
        assert rn.fingerprint(make_settings(over)) == base, over
    # bazt_upstream holds the BAZT_upstream arm's parameters, which the mock platform runs (R3-M1)
    assert rn.fingerprint(make_settings({"bazt_upstream.verdict_probability": 0.9})) != base


def test_a_white_box_cell_finished_with_fewer_evaluation_seeds_is_searched_again_and_keeps_the_old_entries(tmp_path):
    settings = make_settings({"mock.batch_size": 100})
    first = units_of(unit(agents=N, condition="white_box"), (1,), (0,))
    more = units_of(unit(agents=N, condition="white_box"), (1, 2), (0,))
    rn.run_search(first, arm_of, FakePlatform(lambda arm, u: 0.5), settings, tmp_path)
    platform = FakePlatform(lambda arm, u: 0.5)
    table = rn.run_search(more, arm_of, platform, settings, tmp_path)
    assert platform.submitted > 0 and table.missing(more) == [] and table.missing(first) == []
    again = FakePlatform(lambda arm, u: 0.5)
    rn.run_search(more, arm_of, again, settings, tmp_path)
    assert again.calls == []


class Rows:
    def __init__(self):
        self.logs = []

    def log_rows(self, table, rows):
        raise AssertionError("search episodes must not reach the mock_units table")

    def log(self, metrics, step):
        self.logs.append(metrics)


def test_the_platform_gets_a_tracker_that_records_nothing(tmp_path):
    seen = []

    class Spy(FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            seen.append(tracker)
            return super().run_batch(units, arm, settings, store, tracker)
    rec = Rows()
    rn.run_search(units_of(unit(agents=N), (1,), (0,)), arm_of, Spy(lambda arm, u: 0.5), make_settings(), tmp_path,
                  tracker=rec)
    assert seen and all(isinstance(t, core.NullTracker) for t in seen) and rec.logs


def test_refusals_the_runner_preflight_prints_are_core_unset_errors():
    with pytest.raises(core.Unset, match="max_candidates"):
        adaptive.search_episodes(units_of(unit(agents=N), (1,), (0,)), make_settings({"mock.adaptive.max_candidates": 3}))
    with pytest.raises(core.Unset, match="search seeds"):
        adaptive.search_episodes(units_of(unit(agents=N), (8001,), (0,)), make_settings())
    with pytest.raises(core.Unset, match="overlap"):
        adaptive.load_space(make_settings({"mock.adaptive.search_seeds": [20, 40]}))
