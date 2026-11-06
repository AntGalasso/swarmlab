"""D109: the clean reference of an attacked unit is its attack-free twin unit's OWN run. The runner runs the attack-free
units first and attaches unit['twin_id'] to the attacked ones."""
import json

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine import public as mock
from bazt_lab.features.mock_engine.private import platform as P
from bazt_lab.features.mock_engine.private import world as W
from bazt_lab.features.mock_engine.tests.helpers import Fn, make_settings, unit

Actions = core.Actions
CLEAN_COLUMNS = {"accuracy_clean_reference", "recovery_time_accuracy", "nrp", "twin_id"}


def dumps(row):
    return json.dumps(row, sort_keys=True, default=float)


def arm_of(counter=None):
    """A defender whose behaviour depends on its keyed arm stream, so a clean run keyed by another id would differ."""
    def make():
        if counter is not None:
            counter.append(1)

        def fn(obs):
            B, N = obs.active.shape
            draw = torch.stack([core.uniforms(make.ctx.arm_stream(f"x|{obs.t}", b), (N, N)) for b in range(B)])
            return Actions(deny=obs.attempts & (draw > 0.7))
        d = Fn(fn)
        reset = d.reset
        d.reset = lambda ctx: (setattr(make, "ctx", ctx), reset(ctx))[1]
        return d
    return core.ArmSpec("arm", make)


def attacked(seed, attack="infection", twin=True, **kw):
    u = unit(arm="arm", seed=seed, sink=seed % 6, attack=attack, topology="chain", agents=6, audit=0.3, intervention=0.3, **kw)
    if twin:
        u["twin_id"] = twin_of(u)["id"]
    return u


def twin_of(u):
    t = unit(arm="arm", seed=u["seed"], sink=u["sink"], attack="none", topology="chain", agents=6, audit=0.3, intervention=0.3)
    return t


def test_the_clean_accuracy_is_a_float64_ratio_not_float32():
    class T:
        correct = torch.tensor([[[True] * 11 + [False]]])
    got = P.clean_accuracy(T, torch.zeros(1, 12, dtype=torch.bool))
    assert got.dtype == torch.float64 and got.item() == 11 / 12


def test_an_unfinished_twin_is_simulated_with_its_own_keys_saved_as_its_result_and_equals_its_own_run(tmp_path):
    us = [attacked(s) for s in range(1, 5)]
    twins = [twin_of(u) for u in us]
    store = core.CheckpointStore(tmp_path / "a")
    rows = mock.MockPlatform().run_batch(us, arm_of(), make_settings(), store, core.NullTracker())
    assert all(store.is_done(t["id"]) for t in twins)                      # saved as the twins' results
    own = core.CheckpointStore(tmp_path / "b")
    own_rows = mock.MockPlatform().run_batch(twins, arm_of(), make_settings(), own, core.NullTracker())
    saved = [store.rows_of(t["id"])[0] for t in twins]
    assert [dumps(r) for r in saved] == [dumps(r) for r in own_rows]            # bit for bit the twin's own run
    assert [r["accuracy_clean_reference"] for r in rows] == [r["accuracy_clean"] for r in own_rows]


def test_a_finished_twin_is_read_from_the_store_and_no_clean_copy_is_simulated(tmp_path):
    us = [attacked(s) for s in range(1, 5)]
    twins = [twin_of(u) for u in us]
    store = core.CheckpointStore(tmp_path)
    mock.MockPlatform().run_batch(twins, arm_of(), make_settings(), store, core.NullTracker())
    built = []
    rows = mock.MockPlatform().run_batch(us, arm_of(built), make_settings(), store, core.NullTracker())
    assert len(built) == 1                                                  # the attacked run only
    ref = [store.rows_of(t["id"])[0]["accuracy_clean"] for t in twins]
    assert [r["accuracy_clean_reference"] for r in rows] == ref


def test_reading_the_twin_and_simulating_it_give_the_same_attacked_rows(tmp_path):
    us = [attacked(s) for s in range(1, 5)]
    twins = [twin_of(u) for u in us]
    done = core.CheckpointStore(tmp_path / "done")
    mock.MockPlatform().run_batch(twins, arm_of(), make_settings(), done, core.NullTracker())
    read = mock.MockPlatform().run_batch(us, arm_of(), make_settings(), done, core.NullTracker())
    simulated = mock.MockPlatform().run_batch(us, arm_of(), make_settings(), core.CheckpointStore(tmp_path / "fresh"), core.NullTracker())
    assert [dumps(r) for r in read] == [dumps(r) for r in simulated]


def test_rows_match_the_old_path_except_the_clean_reference_columns(tmp_path):
    with_twin = [attacked(s) for s in range(1, 5)]
    without = [{k: v for k, v in u.items() if k != "twin_id"} for u in with_twin]
    a = mock.MockPlatform().run_batch(with_twin, arm_of(), make_settings(), core.CheckpointStore(tmp_path / "a"), core.NullTracker())
    b = mock.MockPlatform().run_batch(without, arm_of(), make_settings(), core.CheckpointStore(tmp_path / "b"), core.NullTracker())
    for x, y in zip(a, b, strict=True):
        assert {k for k in x if dumps(x.get(k)) != dumps(y.get(k))} <= CLEAN_COLUMNS


def test_insider_attacks_take_the_clean_reference_over_the_victims_from_the_twin_keyed_run(tmp_path):
    us = [attacked(s, attack="insider_pi") for s in range(1, 5)]
    store = core.CheckpointStore(tmp_path)
    rows = mock.MockPlatform().run_batch(us, arm_of(), make_settings(), store, core.NullTracker())
    assert all(0.0 <= r["accuracy_clean_reference"] <= 1.0 for r in rows)
    assert all(store.is_done(u["twin_id"]) for u in us)


def test_a_twin_saved_by_an_attacked_unit_is_not_simulated_again_by_a_later_call(tmp_path):
    us = [attacked(s) for s in range(1, 3)]
    twins = [twin_of(u) for u in us]
    store = core.CheckpointStore(tmp_path)
    mock.MockPlatform().run_batch(us, arm_of(), make_settings(), store, core.NullTracker())
    assert mock.MockPlatform().run_batch(twins, arm_of(), make_settings(), store, core.NullTracker()) == []


def grid_pair(seed, attack, **fields):
    """(attacked, twin) the way the runner builds them: the twin is the grid's own attack-free unit (the same grid unit with
    attack 'none', the block's variant fields included) and the attacked unit carries its id as twin_id."""
    base = dict(arm="arm", seed=seed, sink=seed % 6, topology="chain", agents=6, audit=0.3, intervention=0.3, **fields)
    twin = unit(attack="none", **base)
    a = unit(attack=attack, **base)
    a["twin_id"] = twin["id"]
    return a, twin


def adaptive_variant(s):
    """A unit of the adaptive auditor variant of mock_presets_robustness (runner units.preset_variants)."""
    return {"preset": "adaptive", "attack_params": {"auditor_tpr": core.get(s, "auditor_presets.adaptive_tpr")[1]}}


ATTACKED = ("insider_pi", "insider_ta", "insider_ma", "infection", "aitm")


@pytest.mark.parametrize("attack", ATTACKED)
def test_an_attacked_unit_of_the_adaptive_preset_simulates_its_twin_and_saves_the_twins_own_run(tmp_path, attack):
    s = make_settings()
    pairs = [grid_pair(seed, attack, **adaptive_variant(s)) for seed in range(1, 4)]
    store = core.CheckpointStore(tmp_path / "a")
    rows = mock.MockPlatform().run_batch([a for a, _ in pairs], arm_of(), s, store, core.NullTracker())
    own = mock.MockPlatform().run_batch([t for _, t in pairs], arm_of(), s, core.CheckpointStore(tmp_path / "b"),
                                        core.NullTracker())
    saved = [store.rows_of(t["id"]) for _, t in pairs]
    assert all(len(r) == 1 for r in saved)
    assert [dumps(r[0]) for r in saved] == [dumps(r) for r in own]           # bit for bit the grid twin's own run
    if attack not in ("insider_pi", "insider_ta", "insider_ma"):
        assert [r["accuracy_clean_reference"] for r in rows] == [r["accuracy_clean"] for r in own]


@pytest.mark.parametrize("attack", ATTACKED)
def test_an_attacked_unit_of_the_adaptive_preset_runs_when_its_twin_is_already_done(tmp_path, attack):
    s = make_settings()
    pairs = [grid_pair(seed, attack, **adaptive_variant(s)) for seed in range(1, 4)]
    store = core.CheckpointStore(tmp_path)
    own = mock.MockPlatform().run_batch([t for _, t in pairs], arm_of(), s, store, core.NullTracker())
    rows = mock.MockPlatform().run_batch([a for a, _ in pairs], arm_of(), s, store, core.NullTracker())
    assert len(rows) == len(pairs)
    assert [dumps(store.rows_of(t["id"])[0]) for _, t in pairs] == [dumps(r) for r in own]     # the twin row is untouched
    assert all(len(store.rows_of(t["id"])) == 1 for _, t in pairs)
    fresh = mock.MockPlatform().run_batch([a for a, _ in pairs], arm_of(), s, core.CheckpointStore(tmp_path / "fresh"),
                                          core.NullTracker())
    assert [dumps(r) for r in rows] == [dumps(r) for r in fresh]             # reading or simulating the twin: same rows


@pytest.mark.parametrize("attack", ATTACKED)
def test_the_twin_of_an_adaptive_strategy_unit_drops_the_attackers_strategy_like_the_grid_twin(tmp_path, attack):
    """mock_adaptive: the table annotates attacked units with the attacker's strategy and attack-free ones with {}; the
    twin the platform saves must be that attack-free unit, not one that carries the strategy (or no attack_params)."""
    strategy = {"entry_ranks": [1, 2] if attack in W.INSIDERS else [1], "start_frac": 0.4, "rate": 0.5,
                "detectability": 0.5, "potency": 0.75}
    pairs = []
    for seed in range(1, 4):
        a, t = grid_pair(seed, attack)
        a["attack_params"], t["attack_params"] = dict(strategy), {}
        pairs.append((a, t))
    store = core.CheckpointStore(tmp_path / "a")
    mock.MockPlatform().run_batch([a for a, _ in pairs], arm_of(), make_settings(), store, core.NullTracker())
    own = mock.MockPlatform().run_batch([t for _, t in pairs], arm_of(), make_settings(), core.CheckpointStore(tmp_path / "b"),
                                        core.NullTracker())
    assert [dumps(store.rows_of(t["id"])[0]) for _, t in pairs] == [dumps(r) for r in own]


def test_the_twin_is_the_attacked_unit_with_attack_none_its_id_and_only_the_attack_free_parameters():
    a, twin = grid_pair(2, "insider_pi", preset="adaptive", attack_params={"auditor_tpr": 0.3, "start_frac": 0.5,
                                                                           "n_insiders": 2, "potency": 0.5})
    got = P.twin_unit(a)
    assert got == {**twin, "attack_params": {"auditor_tpr": 0.3}}
    assert list(got) == list(twin)                                           # same columns, same order as the grid twin
    b, twin_b = grid_pair(2, "infection")
    assert P.twin_unit(b) == twin_b and "attack_params" not in P.twin_unit(b)


ATTACK_ONLY = {"n_insiders": 3, "insider_bias": 3.0, "mu_inf": 1.0, "entry_ranks": [1], "start_frac": 0.5, "rate": 0.5,
               "detectability": 0.5, "potency": 0.25}


@pytest.mark.parametrize("key", sorted(ATTACK_ONLY))
def test_an_attack_free_run_does_not_depend_on_the_parameters_the_twin_drops(tmp_path, key):
    """The rule behind twin_unit: an attack-free run reads no attack_params key outside ATTACK_FREE_PARAMS."""
    assert set(ATTACK_ONLY) | set(W.ATTACK_FREE_PARAMS) == set(W.PARAM_KEYS) and key not in W.ATTACK_FREE_PARAMS
    plain = [unit(arm="arm", seed=s, sink=s % 6, agents=6, audit=0.3, intervention=0.3) for s in range(1, 4)]
    withp = [{**u, "attack_params": {key: ATTACK_ONLY[key]}} for u in plain]
    a = mock.MockPlatform().run_batch(plain, arm_of(), make_settings(), core.CheckpointStore(tmp_path / "a"), core.NullTracker())
    b = mock.MockPlatform().run_batch(withp, arm_of(), make_settings(), core.CheckpointStore(tmp_path / "b"), core.NullTracker())
    assert [dumps({**r, "attack_params": None}) for r in a] == [dumps({**r, "attack_params": None}) for r in b]


@pytest.mark.parametrize("bad", [None])
def test_units_without_twin_id_keep_the_clean_copy_keyed_by_their_own_id(bad):
    us = [attacked(s, twin=False) for s in range(1, 3)]
    rows = mock.MockPlatform().run_batch(us, arm_of(), make_settings(), core.CheckpointStore(__import__("tempfile").mkdtemp()), core.NullTracker())
    assert all("accuracy_clean_reference" in r for r in rows)
