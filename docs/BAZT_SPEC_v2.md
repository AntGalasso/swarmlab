# BAZT_SPEC_v2 — Budgeted Adaptive Zero-Trust gateway

> **Provenance of this file.** The original `BAZT_SPEC_v2.md` was not in the repository handed over for this branch.
> This document is reconstructed from the implementation brief (patches P1–P8, the 13-module list, the resolved
> decisions) and from the code as built. If the original exists, diff it against §9 (resolved decisions) and keep
> whichever wording is authoritative; the code and `tests/test_bazt_*.py` are the executable reference.
> Everything runs in `--mode mock`; mock numbers are pipeline checks, not findings.

## 1. Problem
A swarm passes free-text status lines between agents. An attacker seeds a payload at the edge; the defender has a
**cost-denominated budget** and can only *see* a noisy monitor. BAZT sits between agents as a gateway: it decides, per
step, what to spend the budget on — checking a node, stripping its channel, cutting an edge, isolating it — trading
contagion avoided against utility lost.

## 2. Attachment to the simulator (patches P1–P8, all default-off)
| id | where | what |
|---|---|---|
| P1 | `EpisodeHooks` | `gateway_factory`, `edge_weight_modifier` (both `None`) |
| P2 | constants | `BAZT_POLICIES` (not in `POLICIES`, `SCALING_POLICIES`, `TUNABLE`) |
| P3 | top of `run_episode` | `gw = factory(...)` only if hooks, factory **and** `policy ∈ BAZT_POLICIES` |
| P4 | inbox + consensus | skip a sender when `not gw.edge_open(j, i, t)`; scale an edge weight by `edge_weight_modifier` |
| P5 | decision | `gw.decide(StepView, truth)`; `plan.isolate → quar`, `plan.sanitize → san_now` |
| P6 | delivery | `restricted(t) ∪ plan.sanitize` are stripped with the existing `u_strip` draw; `gw.on_deliver` |
| P7 | return | `result.update(gw.extra_metrics(infected_series, world))` |
| P8 | `_build_graph` | optional `Config.custom_adj` (validated symmetric, loop-free) |

The classic consensus loop is left verbatim; the guarded variant is a separate helper called only when a gateway or
weight modifier exists. A legacy arm therefore executes exactly the original statements (proved by a SHA-256 over 320
episodes recorded from the original file, `tests/test_bazt_integration.py::test_no_hooks_bit_identical`).

## 3. Information barrier
`StepView(t, active, observed_texts, p_monitor, roll_scores, deg, maxdeg)` — no `infected_now`, `ever` or `world.p0`.
Ground truth reaches the gateway **only** through `gw.inspect_oracle(i, t, truth)`: a paid inspection is conclusive with
probability `q_inspect` (a pre-drawn matrix `u_inspect[t, i]` from CRN stream `bazt-inspect`), otherwise inconclusive and
`truth` is never called. `decoys` is part of the factory signature but never read.

## 4. Evidence
```
ell_{i,t} = a·roll_{i,t} − b + w_prov·tau_{i,t}            p'_i = σ(ell)
```
`a, b` are `swarmlab.Config`'s monitor calibration, so with `w_prov = 0` (or `use_provenance=False`)
`σ(ell) == p_monitor[i]` bit for bit. A conclusive inspection sets `p' = 0.98` (emitting) or `0.02` (clean) for
`inspect_ttl` steps; an inconclusive one leaves `σ(ell)`. An optional `probe_score_hook(i, t, view)` may replace
`roll` (Track-4 activation probes).

## 5. Provenance (lineage taint)
Message ids `(t, src)` with parent links. `tau_{i,1} = 1` for ingress nodes. For `t > 1`
`tau_{i,t} = max(own_decay·tau_{i,t−1}, max_{j→i delivered} delta_hop·tau_eff(j))` with
`tau_eff(j) = (1 − p_strip)·tau_j` if `j`'s message was successfully stripped, else `tau_j`. A clean verdict clears
`tau_i`. Taint after `h` hops is `≤ delta_hop^h`. Taint is built only from delivery events, never from ground truth.

## 6. Actions, pricing, optimizer
`INSPECT, SANITIZE, RESTRICT, BLOCK_EDGE, ISOLATE` (rank = enum value). With `H_i` = forward reach over active nodes and
**open directed edges** (`swarmlab.reach_H`, `min(h_t, dmax)` hops; `h_t = T − t`), `C_i = swarmlab.centrality`,
`lam_eff = lam if use_utility else 0`:

| action | benefit `b` | utility `u` |
|---|---|---|
| ISOLATE | `p'·H_i` | `C_i` |
| RESTRICT | `p_strip·p'·H_i·min(ttl_restrict,h_t)/max(h_t,1)` (+ `priv_bonus·p'·frac` if privileged) | `rho_res·C_i` |
| SANITIZE | `p_strip·p'·H_i/max(h_t,1)` | `rho_san·C_i` |
| BLOCK_EDGE `i→j` | `p'·(1−p'_j)(1 + H_{j∖i}(h_t−1))` | `C_i/max(deg_i,1)` |
| INSPECT | `q_inspect·min(p'·H_i, (1−p')·lam_eff·C_i)` | `0` |

`net = b − lam_eff·(1−p')·u`, `density = net / cost`. For ISOLATE `net = p'H − lam(1−p')C` = `swarmlab.risk_score`.
At `h_t = 0` every reach is 0, so nothing is bought at the last step (also with `use_topology=False`).

`BudgetedOptimizer.plan` is deterministic and RNG-free; ties break on `(−density, node, peer, action_rank)`.
Phase 1 buys affordable inspections with `net > 0, density ≥ θ`, via the oracle, updating `p'`. Phase 2 re-prices, keeps
the best positive-net candidate per target (node, or edge), and accepts greedily in key order while
`density ≥ θ` (stop) and `can_afford` (skip otherwise). θ is `theta0·exp(kappa·spent/capacity)` (`theta0` when
pacing is off). Edge pricing is pruned with an **exact** bound (never changes the plan).

## 7. Budget
Pooled: one pool of `B` cost units (`∞` if `unlimited`). Tiered: hard pool `K_hard·cost_isolate` (ISOLATE, BLOCK_EDGE) and
soft pool `B_soft·cost_sanitize` (INSPECT, SANITIZE, RESTRICT). `charge()` raises `BudgetViolation`; it never clips.

## 8. Arms
Legacy (`no_defense, score_only, centrality_only, risk_aware, risk_aware_tiered`) + `offline_oracle` (bound) +
`static_zta` (always-on role-based send lists + `structured_kv_only` channel from `env_ladder`, no adaptivity, no budget) +
the 7 BAZT arms (`bazt`, and `_no_prov`, `_no_topo`, `_no_pacing`, `_no_util`, `_binary`, `_unlimited`).

## 9. Resolved decisions (where the brief and the repository disagreed or were silent)
1. **Kickoff guard.** The brief names `swarmlab._enforce_kickoff_guard(allow_live_llm)`; the repository's function is
   `swarmlab._kickoff_guard(args)`. The runner exposes `_enforce_kickoff_guard(allow_live_llm)` delegating to it
   unchanged, so `swarmlab.py` needed no edit. Before kickoff `--mode llm` exits unconditionally; afterwards it needs
   `--allow-live-llm`. Provider (`openrouter` | `adaption`) is read from `SWARMLAB_PROVIDER` inside `llm_client`.
2. **Tiered budget units.** Both pools are in cost units (`K_hard` = isolate-equivalents, `B_soft` = sanitize-equivalents),
   so spend is comparable across modes.
3. **`on_deliver` payload.** `{sender: stripped}` for every message that left the gateway; delivered edges are derived from
   the gateway's own send graph, keeping the `run_episode` patch minimal.
4. **Ingress in patient-zero mode** is a CRN-sampled set of `n_ingress` externally-facing nodes (the gateway must not
   know `p0`); with `--entry-source fake_web_jsonl` it is the real set of corpus readers. Provenance therefore only
   carries signal when entry points are known — an honest limit, not a bug.
5. **Isolation lifetime.** `quarantine_ttl = q` keeps a node inactive for `q` full steps after the decision step
   (`None` = permanent, legacy semantics).
6. **Privilege bonus.** Restricting a tool-holder denies its tool authority for the TTL (`priv_bonus`, default 0.5).
7. **`H_edge`** uses the remaining horizon `h_t − 1` (one step is consumed crossing the edge).
8. **`cost_per_prevented_compromise`** needs a counterfactual, so it is paired against the CRN `no_defense` run in the
   runner (`bazt.metrics.add_paired_metrics`), like `swarmlab.add_deltas`, not computed inside the episode.
9. **Filtering cost** lives in `bazt/channel.py` (a 14th, small module beyond the 13 specified, to keep `arms.py` focused):
   see `METRICS_SPEC.md` §15.6.
10. **Extra `BaztConfig` fields** (all defaulted): `own_decay=0.8, n_ingress=2, priv_bonus=0.5, precision_hp=1.0,
    precision_lp=0.6, noisy_frac=0.4`.
11. **Pre-registration guard.** `--split main|heldout` requires `--confirm-frozen-config`; the default output name for the
    tune split is `smoke_tune_check.csv`.

## 10. Known behaviour of the untuned defaults (mock, tune seeds 9001–9012; not a finding)
With the specified defaults the gateway front-loads its budget: in **all 48** decoy-regime tune episodes at least 10 of
the 12 cost units are spent at step 1, and over the 96-row smoke run 54 % of the spend is inspection, 27 % edge blocks,
17 % isolation, ~1 % sanitising, 0 % restriction. The EVPI term stays positive for any node with `p' > 0`, and `theta0`
is low while the pool is full, so the first step buys everything it can. On that run `bazt` does not beat `risk_aware`
on `AUC_ever` / `MAE`, whereas `bazt_binary` and `bazt_unlimited` do. Tuning `theta0 / kappa / q_inspect` (and possibly a
floor on `p'` for inspection) on the **tune** split is the obvious next step; main/heldout stay untouched until the
config is frozen.
