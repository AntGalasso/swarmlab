"""Integration tests for BAZT end-to-end via `swarmlab.run_episode` with
`EpisodeHooks.gateway_factory`. Offline, `--mode mock`, no API key."""
import swarmlab as sl

from bazt.arms import bazt_config
from bazt.gateway import EnforcementGateway
from bazt.roles import assign_roles

BAZT_POLICIES = sl.BAZT_POLICIES


def _make_hooks(bcfg_name="bazt", N=6, seed=9001, n_decoys=3):
    cfg = sl.Config(N=N)
    w = sl.make_world("small_world", seed, cfg)
    decoys = sl.pick_decoys(w, n_decoys)
    base_adj = w.adj
    roles = assign_roles(N, w.topology, seed, base_adj, set(), 2)
    ingress = {r.node_id for r in roles if r.role == "ingress"}
    gw = EnforcementGateway(cfg, bazt_config(bcfg_name), w.topology, seed,
                            base_adj, roles, ingress, ingress)

    def factory(policy, world2, cfg2, base_adj2, decoys2, hooks2):
        if policy in BAZT_POLICIES:
            return gw
        return None

    return sl.EpisodeHooks(gateway_factory=factory), cfg, w, decoys


def test_bazt_episode_runs_and_returns_section15_keys():
    hooks, cfg, w, decoys = _make_hooks()
    r = sl.run_episode(w, "bazt", cfg, sl.MockAgentPolicy(cfg), decoys, hooks=hooks)
    assert "task_mae" in r and "auc_ever_infected" in r
    assert "enforcement_spend" in r
    assert "bazt_budget_spent" in r
    assert "bazt_budget_capacity" in r
    assert "bazt_budget_violations" in r
    assert r["bazt_budget_violations"] == 0.0
    assert r["quarantines_used"] >= 0
    assert r["sanitizations_used"] >= 0


def test_all_bazt_ablation_policies_run():
    hooks, cfg, w, decoys = _make_hooks()
    for policy in BAZT_POLICIES:
        r = sl.run_episode(w, policy, cfg, sl.MockAgentPolicy(cfg), decoys, hooks=hooks)
        assert r["task_mae"] is not None
        assert r["bazt_budget_violations"] == 0.0


def test_bazt_binary_actions_only_isolate():
    hooks, cfg, w, decoys = _make_hooks(bcfg_name="bazt_binary")
    r = sl.run_episode(w, "bazt_binary", cfg, sl.MockAgentPolicy(cfg), decoys, hooks=hooks)
    assert r["bazt_budget_violations"] == 0.0
    assert r["task_mae"] is not None


def test_bazt_unlimited_no_capacity():
    hooks, cfg, w, decoys = _make_hooks(bcfg_name="bazt_unlimited")
    r = sl.run_episode(w, "bazt_unlimited", cfg, sl.MockAgentPolicy(cfg), decoys, hooks=hooks)
    assert r["bazt_budget_capacity"] == -1.0
    assert r["bazt_budget_violations"] == 0.0


def test_bazt_heldout_seeds_deterministic():
    results = []
    for seed in (9001, 9002, 9003):
        hooks, cfg, w, decoys = _make_hooks(seed=seed)
        r = sl.run_episode(w, "bazt", cfg, sl.MockAgentPolicy(cfg), decoys, hooks=hooks)
        results.append((r["task_mae"], r["auc_ever_infected"], r["enforcement_spend"]))
    # same seed -> same result (run twice)
    hooks, cfg, w, decoys = _make_hooks(seed=9001)
    r2 = sl.run_episode(w, "bazt", cfg, sl.MockAgentPolicy(cfg), decoys, hooks=hooks)
    assert (r2["task_mae"], r2["auc_ever_infected"], r2["enforcement_spend"]) == results[0]


def test_non_bazt_policy_gateway_none():
    hooks, cfg, w, decoys = _make_hooks()
    r = sl.run_episode(w, "no_defense", cfg, sl.MockAgentPolicy(cfg), decoys, hooks=hooks)
    # no gateway for non-BAZT policy -> no BAZT keys in result
    assert "bazt_budget_spent" not in r
    assert r["quarantines_used"] == 0
    assert r["sanitizations_used"] == 0
