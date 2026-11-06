"""The search against the real mock platform (mock_engine.public.MockPlatform), not a fake.

What this pins: every strategy the space can produce is accepted by the platform for every attack family, a strategy
re-run on its own episode reproduces the ASR the search saw (common random numbers across the two features), and the
two attack-parameter protocols agree (entry ranks, schedule, detectability, potency).

Every number in `engine_settings` is a TEST value that makes the scenario small. None is an experiment decision.
"""
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.adaptive import public as adaptive
from bazt_lab.features.mock_engine import public as mock_engine

FAMILIES = ("insider_pi", "insider_ta", "insider_ma", "infection", "aitm")
N = 6


def engine_settings(**over) -> dict:
    s = core.load_settings(profile="sprint")
    values = {
        "runtime.n_insiders": 2, "runtime.rate_limit_per_identity": 1, "runtime.horizon_constant": 3.0,
        "runtime.eps_correct": 2.0, "runtime.ba_m": 2, "runtime.ws_k": 4, "runtime.ws_rewire_p": 0.25,
        "runtime.sanitize_success": 1.0, "runtime.insider_bias": 8.0, "runtime.ta_detectability": 1.5,
        "runtime.ma_detectability": 1.2, "runtime.aitm_rewrite_strength": 1.0, "runtime.recovery_margin": 0.05,
        "mock.model.bias": 8.0, "mock.model.burst_span": 1, "mock.auditor.binormal_method": "shared_threshold",
        "costs.main.deny": 1, "costs.robustness.deny": 1, "mock.batch_size": 64,
        "mock.adaptive.search_seeds": [8001, 8002], "mock.adaptive.insider_sets": "contiguous",
        "mock.adaptive.start_fractions": [0.0, 0.5], "mock.adaptive.rates": [1.0, 0.5],
        "mock.adaptive.detectabilities": [1.0, 0.5], "mock.adaptive.evasion_cost": 0.5,
        "mock.adaptive.max_candidates": 1000, "mock.adaptive.conditions": list(adaptive.CONDITIONS),
        "mock.adaptive.knowledge.gray_box": "cell", "mock.adaptive.knowledge.white_box": "episode",
        "mock.adaptive.knowledge.embedding_evasion": "cell", "mock.adaptive.knowledge.attack_rate": "cell",
    }
    for key, value in {**values, **{k.replace("__", "."): v for k, v in over.items()}}.items():
        core.put(s, key, value)
    return s


class IsolateFlagged:
    """A tiny defender: isolates every active agent whose free monitor probability exceeds 0.8."""
    name = "isolate_flagged"

    def reset(self, ctx):
        pass

    def decide(self, obs):
        return core.Actions(isolation=obs.active & (obs.monitor > 0.8))

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


ARM = core.ArmSpec(name="isolate_flagged", factory=IsolateFlagged)


def arm_of(name):
    assert name == ARM.name
    return ARM


def eval_units(family, condition, seeds=(1, 2), sinks=(0, 3), topology="small_world", audit=0.1, intervention="uncapped"):
    return [dict(block="mock_adaptive", platform="mock", arm=ARM.name, audit=audit, intervention=intervention,
                 topology=topology, attack=family, agents=N, seed=s, sink=k, condition=condition)
            for s in seeds for k in sinks]


def with_ids(units):
    return [{**u, "id": f"{u['condition']}-{u['attack']}-{u['seed']}-{u['sink']}", "thresholds": {}} for u in units]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_strategy_of_every_condition_is_accepted_by_the_platform_for_every_family(family, tmp_path):
    settings = engine_settings()
    for condition in adaptive.CONDITIONS:
        units = eval_units(family, condition, seeds=(1,), sinks=(0,))
        table = adaptive.run_search(units, arm_of, mock_engine.MockPlatform(), settings, tmp_path / condition)
        choice = table.choice(units[0])
        assert choice.candidates == adaptive.candidate_count(family, N, condition, adaptive.load_space(settings))
        assert choice.asr >= choice.baseline_asr                      # the family default is a candidate


@pytest.mark.parametrize("family", FAMILIES)
def test_a_white_box_strategy_rerun_on_its_episode_reproduces_the_searched_asr(family, tmp_path):
    settings = engine_settings()
    units = eval_units(family, "white_box")
    table = adaptive.run_search(units, arm_of, mock_engine.MockPlatform(), settings, tmp_path)
    annotated = with_ids([table.annotate(u) for u in units])
    rows = mock_engine.MockPlatform().run_batch(annotated, ARM, settings, core.CheckpointStore(tmp_path / "eval"),
                                                core.NullTracker())
    assert len(rows) == len(units)
    for u, row in zip(annotated, rows, strict=True):
        assert row["asr_final"] == table.choice(u).asr          # exact: one episode per white-box entry


def test_the_best_response_never_does_worse_than_the_default_attack_on_the_evaluation_episodes(tmp_path):
    """A white-box attacker plays the best of a space that holds the default, on the very episodes it is scored on."""
    settings = engine_settings()
    units = eval_units("infection", "white_box", seeds=(1, 2, 3), sinks=(0, 2, 4), topology="star_leaf")
    table = adaptive.run_search(units, arm_of, mock_engine.MockPlatform(), settings, tmp_path)
    plain = mock_engine.MockPlatform().run_batch(with_ids(units), ARM, settings, core.CheckpointStore(tmp_path / "a"),
                                                 core.NullTracker())
    best = mock_engine.MockPlatform().run_batch(with_ids([table.annotate(u) for u in units]), ARM, settings,
                                                core.CheckpointStore(tmp_path / "b"), core.NullTracker())
    for p, b in zip(plain, best, strict=True):
        assert b["asr_final"] >= p["asr_final"]
    assert sum(b["asr_final"] for b in best) > sum(p["asr_final"] for p in plain)      # a random leaf is not the best entry


def test_a_gray_box_cell_has_one_strategy_searched_on_the_search_seeds_with_every_sink(tmp_path):
    settings = engine_settings()
    units = eval_units("infection", "gray_box", seeds=(1, 2), sinks=(0,))
    table = adaptive.run_search(units, arm_of, mock_engine.MockPlatform(), settings, tmp_path)
    assert len({str(table.params_for(u)) for u in units}) == 1
    assert table.choice(units[0]).episodes == 2 * N                     # 2 search seeds x every sink
