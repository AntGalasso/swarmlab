"""End-to-end BAZT contract: legacy bit-identity, ground-truth firewall, budgets, filtering cost, kickoff guard,
dependency direction, result-file protection. Mock mode only: this file performs ZERO live LLM calls."""
import ast
import glob
import hashlib
import json
import os
import re

import numpy as np
import pytest

from bazt_testkit import TUNE
import swarmlab as sl
from bazt import runner
from bazt.arms import BAZT_ARMS, DEFAULT_ARMS, STATIC_ARM, RunContext, equalized_swarm_cfg, build_episode
from bazt.channel import RationaleAgent, make_edge_weight_modifier, sensor_classes
from bazt.config import BaztConfig
from bazt.gateway import EnforcementGateway
from bazt.metrics import BAZT_METRICS
from bazt.optimizer import Plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUTURE, PAST = "2999-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00"
# sha256 over every legacy episode result (4 topologies x seeds 9001-9004 x 2 regimes x 10 policies), recorded
# from the ORIGINAL swarmlab.py before any BAZT patch existed.
LEGACY_GOLDEN = "1caebe365f618714f3dd12b7d797f8f6ecb3bf2fe5417a5eafad9da46ee0faa0"


def _ctx(**kw):
    bcfg = kw.pop("bcfg", BaztConfig())
    return RunContext(bcfg=bcfg, agent_factory=lambda cfg, world: sl.MockAgentPolicy(cfg), **kw)


def _legacy_results(hooks_for=lambda: None):
    cfg, out = sl.Config(), []
    for topo in sl.TOPOLOGIES:
        for seed in range(9001, 9005):
            w = sl.make_world(topo, seed, cfg)
            for regime in ("clean", "decoy"):
                dec = sl.pick_decoys(w, 3 if regime == "decoy" else 0)
                for pol in sl.POLICIES:
                    out.append(sl.run_episode(w, pol, cfg, sl.MockAgentPolicy(cfg), dec, hooks=hooks_for()))
    return out


def _digest(results):
    h = hashlib.sha256()
    for r in results:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()


# ===================================================================================== legacy contract
def test_no_hooks_bit_identical():
    assert _digest(_legacy_results()) == LEGACY_GOLDEN


def test_default_hooks_and_neutral_weight_modifier_are_bit_identical():
    base = _legacy_results()
    plain = _legacy_results(lambda: sl.EpisodeHooks())
    unit = _legacy_results(lambda: sl.EpisodeHooks(edge_weight_modifier=lambda j, i, t, m: 1.0))
    strip = lambda rs: [{k: v for k, v in r.items() if not k.startswith(("constraint_", "message_"))} for r in rs]
    assert strip(plain) == base and strip(unit) == base


def test_legacy_arms_bit_identical_with_gateway_factory_present():
    def boom(*a, **k):
        raise AssertionError("the gateway factory must never be consulted for a legacy policy")
    base = _legacy_results()
    with_factory = _legacy_results(lambda: sl.EpisodeHooks(gateway_factory=boom))
    drop = lambda rs: [{k: v for k, v in r.items() if not k.startswith(("constraint_", "message_"))} for r in rs]
    assert drop(with_factory) == base


def test_bazt_policy_without_a_factory_is_refused_not_silently_legacy():
    w = sl.make_world("chain", 9001, sl.Config())
    for hooks in (None, sl.EpisodeHooks()):
        with pytest.raises(ValueError):
            sl.run_episode(w, "bazt", sl.Config(), sl.MockAgentPolicy(sl.Config()), frozenset(), hooks=hooks)


def test_swarmlab_registries_and_metrics_are_untouched():
    assert sl.BAZT_POLICIES == ("bazt", "bazt_no_prov", "bazt_no_topo", "bazt_no_pacing", "bazt_no_util",
                                "bazt_binary", "bazt_unlimited")
    assert not set(sl.BAZT_POLICIES) & (set(sl.POLICIES) | set(sl.SCALING_POLICIES) | set(sl.TUNABLE))
    assert not set(BAZT_METRICS) & set(sl.METRICS + sl.EXTRA_COLS)         # swarmlab.METRICS was NOT extended
    assert len(sl.POLICIES) == 10 and len(sl.SCALING_POLICIES) == 5


def test_custom_adjacency_is_validated_and_named_topologies_are_unaffected():
    ring = tuple(tuple(sorted(((i - 1) % 6, (i + 1) % 6))) for i in range(6))
    w = sl.make_world("custom", 1, sl.Config(N=6, T=6, custom_adj=ring))
    assert w.adj == [list(a) for a in ring]
    with pytest.raises(ValueError):
        sl.make_world("custom", 1, sl.Config(N=6, custom_adj=((1,), (), (), (), (), ())))     # asymmetric
    with pytest.raises(ValueError):
        sl.make_world("custom", 1, sl.Config(N=5, custom_adj=ring))                            # wrong size
    assert sl.make_world("chain", 3, sl.Config()).adj == sl.make_world("chain", 3, sl.Config(custom_adj=None)).adj


# ======================================================================================= firewall
class SpyGateway(EnforcementGateway):
    views, reads, conclusive = [], [], []

    def decide(self, view, truth):
        SpyGateway.views.append(view)
        plan = super().decide(view, lambda i: SpyGateway.reads.append(i) or truth(i))
        SpyGateway.conclusive += [i for i, v in plan.inspected.items() if v is not None]
        return plan


def _spy_run(bcfg, seed=9001, topo="small_world"):
    SpyGateway.views, SpyGateway.reads, SpyGateway.conclusive = [], [], []

    def factory(policy, world, cfg, base_adj, decoys, hooks):
        return SpyGateway(policy, bcfg, cfg, world.topology, world.seed, base_adj, base_adj, frozenset())
    w, cfg = sl.make_world(topo, seed, sl.Config()), sl.Config()
    dec = sl.pick_decoys(w, 3)
    return sl.run_episode(w, "bazt", cfg, sl.MockAgentPolicy(cfg), dec, hooks=sl.EpisodeHooks(gateway_factory=factory))


def test_ground_truth_firewall():
    # 1. the view has no ground-truth field
    assert set(sl.StepView.__dataclass_fields__) == {"t", "active", "observed_texts", "p_monitor", "roll_scores",
                                                     "deg", "maxdeg"}
    _spy_run(BaztConfig())
    assert len(SpyGateway.views) == sl.Config().T
    for v in SpyGateway.views:
        assert not any(hasattr(v, f) for f in ("infected_now", "ever", "p0", "world"))
    # 2. truth is read ONLY by paid, conclusive inspections
    assert SpyGateway.reads and sorted(SpyGateway.reads) == sorted(SpyGateway.conclusive)
    # 3. with no conclusive inspection possible the episode completes without ever touching ground truth
    SpyGateway.reads.clear()
    r = _spy_run(BaztConfig(q_inspect=0.0))
    assert SpyGateway.reads == [] and r["inspections_conclusive"] == 0
    # 4. static guarantee: no module outside metrics.py mentions the hidden state
    for path in glob.glob(os.path.join(ROOT, "bazt", "*.py")):
        if os.path.basename(path) in ("metrics.py", "runner.py"):
            continue
        src = re.sub(r'""".*?"""', "", open(path).read(), flags=re.S)
        assert not re.search(r"\.p0\b|infected_now|\.ever\b|\bever\[", src), path


def test_gateway_decisions_do_not_depend_on_decoys():
    cfg = sl.Config()
    w = sl.make_world("chain", 9001, cfg)
    factory = __import__("bazt.arms", fromlist=["x"]).make_gateway_factory(_ctx())
    h = sl.EpisodeHooks()
    a = factory("bazt", w, cfg, w.adj, frozenset(), h)
    b = factory("bazt", w, cfg, w.adj, frozenset({1, 2, 3}), h)
    assert a.identities == b.identities and (a._u_inspect == b._u_inspect).all()


# ======================================================================================== budgets
def test_budget_violations_zero_all_seeds():
    arms = list(BAZT_ARMS)
    rows, _ = runner.run_bazt_grid(sl.TOPOLOGIES, ["clean", "decoy"], arms, TUNE, sl.Config(), _ctx(), progress=False)
    assert len(rows) == len(sl.TOPOLOGIES) * 2 * len(TUNE) * len(arms)
    assert all(r["budget_violations"] == 0 for r in rows)
    assert all(r["enforcement_spend"] <= 12 for r in rows if r["policy"] != "bazt_unlimited")
    assert any(r["enforcement_spend"] > 12 for r in rows if r["policy"] == "bazt_unlimited")
    for r in rows:
        parts = sum(r[k] for k in ("spend_inspect", "spend_sanitize", "spend_restrict", "spend_block", "spend_isolate"))
        assert parts == r["enforcement_spend"]


def test_tiered_budget_respects_both_pools():
    ctx = _ctx(bcfg=BaztConfig(budget_mode="tiered", K_hard=1, B_soft=3))
    rows, _ = runner.run_bazt_grid(sl.TOPOLOGIES, ["decoy"], ["bazt"], TUNE[:6], sl.Config(), ctx, progress=False)
    for r in rows:
        assert r["budget_violations"] == 0
        assert r["spend_isolate"] + r["spend_block"] <= 3 and r["spend_inspect"] + r["spend_sanitize"] + r["spend_restrict"] <= 3


def test_nothing_is_spent_at_the_last_step():
    ctx = _ctx()
    ctx.sink = []
    for seed in TUNE[:4]:
        w = sl.make_world("small_world", seed, sl.Config())
        s = build_episode("bazt", w, sl.Config(), ctx)
        sl.run_episode(w, s.policy, s.cfg, s.agent, sl.pick_decoys(w, 3), hooks=s.hooks)
        gw = ctx.sink.pop()
        assert gw.audit.rows() and max(r["t"] for r in gw.audit.rows()) < sl.Config().T


def test_equalized_budget_gives_legacy_arms_the_same_total_cost():
    b, cfg = BaztConfig(B=12), sl.Config()
    for arm in ("no_defense", "score_only", "centrality_only", "risk_aware", "offline_oracle"):
        assert equalized_swarm_cfg(arm, cfg, b).K == 4                   # 12 // cost_isolate
    t = equalized_swarm_cfg("risk_aware_tiered", cfg, b)
    assert (t.K, t.B_san) == (2, 6) and t.K * 3 + t.B_san * 1 <= 12
    assert equalized_swarm_cfg("risk_aware", cfg, BaztConfig(B=7)).K == 2
    tt = equalized_swarm_cfg("risk_aware_tiered", cfg, BaztConfig(budget_mode="tiered", K_hard=2, B_soft=4))
    assert (tt.K, tt.B_san) == (2, 4)
    assert equalized_swarm_cfg("bazt", cfg, b) is cfg and equalized_swarm_cfg(STATIC_ARM, cfg, b) is cfg


def test_equalized_legacy_arms_never_exceed_the_budget_in_cost_units():
    rows, _ = runner.run_bazt_grid(sl.TOPOLOGIES, ["decoy"], ["risk_aware", "risk_aware_tiered", "score_only"],
                                   TUNE[:6], sl.Config(), _ctx(equalize=True), progress=False)
    assert all(r["enforcement_spend"] <= 12 for r in rows)


# =================================================================================== filtering cost
class StripAllGateway:
    """Restricts every node at every step and buys nothing else: isolates the price of filtering itself."""

    def __init__(self, n):
        self.n = n

    edge_open = staticmethod(lambda j, i, t=0: True)
    pop_released = staticmethod(lambda t: [])
    on_deliver = staticmethod(lambda t, out: None)
    extra_metrics = staticmethod(lambda series, world=None: {})

    def decide(self, view, truth):
        return Plan()

    def restricted(self, t):
        return frozenset(range(self.n))


def _clean_mae(strip_all: bool, weigh: bool):
    """MAE with NO infected node at all (entry suppressed), so only the filtering effect is measured."""
    cfg, bcfg, out = sl.Config(), BaztConfig(), []
    mod = make_edge_weight_modifier(bcfg) if weigh else None
    for topo in sl.TOPOLOGIES:
        for seed in TUNE:
            w = sl.make_world(topo, seed, cfg)
            agent = RationaleAgent(sl.MockAgentPolicy(cfg), sensor_classes(topo, cfg.N, seed, bcfg.noisy_frac))
            hooks = sl.EpisodeHooks(suppress_patient_zero=True, edge_weight_modifier=mod,
                                    gateway_factory=(lambda *a: StripAllGateway(cfg.N)) if strip_all else None)
            out.append(sl.run_episode(w, "bazt" if strip_all else "no_defense", cfg, agent, frozenset(),
                                      hooks=hooks)["task_mae"])
    return np.array(out)


def test_strip_weight_loss_increases_clean_mae():
    stripped, intact = _clean_mae(True, True), _clean_mae(False, True)
    assert stripped.mean() > 1.15 * intact.mean()                          # a real, not marginal, price
    assert (stripped > intact).mean() > 0.75                               # and it shows up episode by episode
    # without the weight rule stripping is (almost) free -- the legacy behaviour this feature fixes
    free_s, free_i = _clean_mae(True, False), _clean_mae(False, False)
    assert abs(free_s.mean() - free_i.mean()) < 0.01 * free_i.mean()
    assert stripped.mean() > free_s.mean()


def test_disabling_the_filtering_cost_removes_rationale_and_weights():
    ctx = _ctx(bcfg=BaztConfig(strip_degrades_weight=False))
    s = build_episode("bazt", sl.make_world("chain", 9001, sl.Config()), sl.Config(), ctx)
    assert s.hooks.edge_weight_modifier is None and not isinstance(s.agent, RationaleAgent)


# ============================================================================== arms and the runner
def test_all_fourteen_arms_run_and_report_every_section_15_column():
    rows, _ = runner.run_bazt_grid(["small_world"], ["decoy"], DEFAULT_ARMS, TUNE[:2], sl.Config(), _ctx(), progress=False)
    runner.finalize(rows)
    assert {r["policy"] for r in rows} == set(DEFAULT_ARMS) and len(DEFAULT_ARMS) == 14
    for r in rows:
        assert set(BAZT_METRICS) <= set(r) and set(sl.METRICS) - {"delta_abs", "delta_rel"} <= set(r)
        assert 0.0 <= r["auc_ever_infected"] <= 1.0
    bz = [r for r in rows if r["policy"] == "bazt"]
    assert all(0.0 <= r["message_throughput"] <= 1.0 and 0.0 <= r["harm_exposure_auc"] <= 1.0 for r in bz)


def test_grid_is_deterministic_and_arm_results_do_not_depend_on_the_other_arms():
    a, _ = runner.run_bazt_grid(["chain"], ["decoy"], list(DEFAULT_ARMS), TUNE[:2], sl.Config(), _ctx(), progress=False)
    b, _ = runner.run_bazt_grid(["chain"], ["decoy"], ["risk_aware", "bazt"], TUNE[:2], sl.Config(), _ctx(), progress=False)
    c, _ = runner.run_bazt_grid(["chain"], ["decoy"], list(DEFAULT_ARMS), TUNE[:2], sl.Config(), _ctx(), progress=False)
    def same(x, y):
        return json.dumps(x, sort_keys=True, default=str) == json.dumps(y, sort_keys=True, default=str)
    assert same(a, c)
    pick = lambda rows, arm: [r for r in rows if r["policy"] == arm]
    for arm in ("risk_aware", "bazt"):
        assert same(pick(a, arm), pick(b, arm))


def test_static_zta_uses_role_edges_and_seals_the_free_text_channel():
    w = sl.make_world("small_world", 9001, sl.Config())
    s = build_episode(STATIC_ARM, w, sl.Config(), _ctx())
    assert s.policy == "no_defense" and s.hooks.recv_from is not None
    assert s.hooks.channel_filter("a1 t2 est=3.0 [REF-7] sens=hp", 1, 2) == "a1 t2 est=3.0"
    from bazt.roles import assign_roles
    ids = assign_roles(w.topology, w.seed, w.adj, BaztConfig(), frozenset())
    for v, senders in enumerate(s.hooks.recv_from):
        assert all(v in ids[u].allowed_out_edges for u in senders)
    r = sl.run_episode(w, s.policy, s.cfg, s.agent, frozenset(), hooks=s.hooks)
    assert r["auc_ever_infected"] == 0.0                                     # the payload cannot ride a schema-only channel


def test_quarantine_ttl_releases_nodes_inside_run_episode():
    ctx = _ctx(bcfg=BaztConfig(quarantine_ttl=2))
    ctx.sink = []
    seen = False
    for seed in TUNE:
        w = sl.make_world("chain", seed, sl.Config())
        s = build_episode("bazt", w, sl.Config(), ctx)
        sl.run_episode(w, s.policy, s.cfg, s.agent, sl.pick_decoys(w, 3), hooks=s.hooks)
        seen |= any(r["reason"] == "quarantine_ttl" for r in ctx.sink.pop().audit.rows())
    assert seen


def test_evasion_adapter_and_constraint_hooks_compose_with_the_gateway():
    for name in ("naive", "compressed", "live_cot"):
        out = runner.main(["--mode", "mock", "--split", "tune", "--seeds", "1", "--topologies", "small_world",
                           "--regimes", "decoy", "--arms", "no_defense", "bazt", "--evasion-adapter", name,
                           "--channel-mode", "banned_brackets", "--out", os.path.join(_TMP[0], name), "--quiet"])
        assert out == 0


_TMP = [None]


@pytest.fixture(autouse=True)
def _tmpdir(tmp_path):
    _TMP[0] = str(tmp_path)


# ================================================================================ runner / compliance
def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def test_runner_writes_only_under_its_out_dir_and_never_touches_legacy_results(tmp_path):
    legacy = [os.path.join(ROOT, "results", f) for f in ("per_seed_main.csv", "per_seed_heldout.csv")]
    before = [_md5(p) for p in legacy if os.path.exists(p)]
    out = tmp_path / "bazt"
    assert runner.main(["--mode", "mock", "--split", "tune", "--seeds", "2", "--topologies", "chain",
                        "--regimes", "decoy", "--out", str(out), "--write-audit", "--quiet"]) == 0
    assert sorted(os.listdir(out)) == ["smoke_tune_check.audit.csv", "smoke_tune_check.csv", "smoke_tune_check.meta.json"]
    meta = json.load(open(out / "smoke_tune_check.meta.json"))
    assert meta["mode"] == "mock" and meta["seeds"] == [9001, 9002] and "probe_score_hook" not in meta["bazt_config"]
    header = open(out / "smoke_tune_check.csv").readline().strip().split(",")
    assert set(BAZT_METRICS) <= set(header) and set(sl.METRICS) <= set(header)
    assert [_md5(p) for p in legacy if os.path.exists(p)] == before


def test_main_and_heldout_need_an_explicit_frozen_config_certificate(tmp_path):
    for split in ("main", "heldout"):
        with pytest.raises(SystemExit):
            runner.main(["--mode", "mock", "--split", split, "--seeds", "1", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        runner.main(["--mode", "mock", "--arms", "nonsense", "--out", str(tmp_path)])


def test_budget_sweep_is_cost_matched_and_writes_one_file(tmp_path):
    out = tmp_path / "bazt"
    assert runner.main(["--mode", "mock", "--split", "tune", "--seeds", "2", "--topologies", "chain", "--regimes", "decoy",
                        "--arms", "no_defense", "risk_aware", "bazt", "--budget-sweep", "--sweep-B", "3", "12",
                        "--out", str(out), "--quiet"]) == 0
    import csv
    rows = list(csv.DictReader(open(out / "budget_sweep_tune.csv")))
    assert {r["B"] for r in rows} == {"3", "12"}
    ra = {r["B"]: max(float(x["quarantines_used"]) for x in rows if x["policy"] == "risk_aware" and x["B"] == r["B"]) for r in rows}
    assert ra["3"] <= 1 and ra["12"] <= 4                                      # K = B // cost_isolate


def test_kickoff_guard_blocks_live_llm_before_kickoff(monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", FUTURE)
    monkeypatch.setattr(runner, "_make_backend", lambda a: pytest.fail("a backend must not even be built before kickoff"))
    for provider in ("openrouter", "adaption"):
        monkeypatch.setenv("SWARMLAB_PROVIDER", provider)
        for extra in ([], ["--allow-live-llm"], ["--allow-live-llm", "--evasion-adapter", "live_cot"]):
            with pytest.raises(SystemExit):
                runner.main(["--mode", "llm", "--split", "tune", "--seeds", "1", "--out", str(tmp_path)] + extra)
    with pytest.raises(SystemExit):
        runner._enforce_kickoff_guard(True)
    assert not os.listdir(tmp_path)                                            # nothing written, nothing built


def test_llm_mode_needs_the_flag_even_after_kickoff(monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    with pytest.raises(SystemExit):
        runner.main(["--mode", "llm", "--split", "tune", "--out", str(tmp_path)])


class FakeCache:
    def stats(self):
        return {"live_calls": 0}

    def close(self):
        pass


class FakeBackend:
    """Stands in for llm_client.LLMBackend after kickoff: no network, no key."""
    cache = FakeCache()

    def __init__(self):
        self.calls = 0

    def generate(self, reqs):
        self.calls += len(reqs)
        return [f"status est={50.0 + k:.1f}" for k, _ in enumerate(reqs)]

    def close(self):
        pass


@pytest.mark.parametrize("provider", ["openrouter", "adaption"])
def test_llm_mode_runs_after_kickoff_without_code_edits(monkeypatch, tmp_path, provider):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    monkeypatch.setenv("SWARMLAB_PROVIDER", provider)
    fake = FakeBackend()
    monkeypatch.setattr(runner, "_make_backend", lambda a: fake)
    assert runner.main(["--mode", "llm", "--allow-live-llm", "--split", "tune", "--seeds", "1", "--topologies", "chain",
                        "--regimes", "decoy", "--arms", "no_defense", "bazt", "--out", str(tmp_path), "--quiet"]) == 0
    assert fake.calls == 2 * sl.Config().N * sl.Config().T                     # one status line per agent-step, per arm


def test_live_cot_adapter_hook_is_wired_through_the_llm_path(monkeypatch, tmp_path):
    import live_cot_adapter as lca
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    fake, compress = FakeBackend(), FakeBackend()
    monkeypatch.setattr(runner, "_make_backend", lambda a: fake)
    monkeypatch.setattr(lca, "_make_backend", lambda: compress)
    assert runner.main(["--mode", "llm", "--allow-live-llm", "--evasion-adapter", "live_cot", "--split", "tune",
                        "--seeds", "1", "--topologies", "chain", "--regimes", "decoy", "--arms", "bazt",
                        "--out", str(tmp_path), "--quiet"]) == 0
    assert compress.calls > 0                                                  # the infected path compressed through the adapter


# ====================================================================================== dependencies
def _imports(path):
    tree = ast.parse(open(path).read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_dependency_direction():
    for legacy in ("swarmlab.py", "env_ladder.py", "llm_client.py", "live_cot_adapter.py", "stats_check.py"):
        assert "bazt" not in _imports(os.path.join(ROOT, legacy)), legacy
    assert "llm_client" not in _imports(os.path.join(ROOT, "env_ladder.py"))
    for path in glob.glob(os.path.join(ROOT, "bazt", "*.py")):
        tree = ast.parse(open(path).read())
        top_level = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top_level |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        assert not top_level & {"llm_client", "live_cot_adapter"}, f"{path}: live modules are imported lazily only"
    allowed = {"swarmlab", "env_ladder", "bazt", "numpy", "llm_client", "live_cot_adapter"}
    for path in glob.glob(os.path.join(ROOT, "bazt", "*.py")):
        third = _imports(path) - allowed - set(__import__("sys").stdlib_module_names)
        assert not third, (path, third)


def test_legacy_core_test_file_is_untouched():
    """Contract: tests/test_core.py is never edited by the BAZT branch (md5 recorded before the branch)."""
    assert _md5(os.path.join(ROOT, "tests", "test_core.py")) == "7bf5008a0ee65bb745604d2a2110bc61"
