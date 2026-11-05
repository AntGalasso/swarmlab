# BAZT — architecture for the repository

`feat/bazt` adds a Budgeted Adaptive Zero-Trust gateway **on top of** the simulator. It is written before kickoff and
declared as **pre-existing scaffolding**; it contains no experimental result (spec: `docs/BAZT_SPEC_v2.md`).

## Dependency rule
```text
swarmlab.py ◄── env_ladder.py ◄── bazt/*          (arrows = "is imported by")
llm_client.py, live_cot_adapter.py ◄── bazt/runner.py   (lazy, inside functions only)
```
`swarmlab.py` and `env_ladder.py` never import `bazt`; `env_ladder.py` never imports `llm_client`. Enforced by
`tests/test_bazt_integration.py::test_dependency_direction`.

## Modules (`bazt/`)
```text
config ─► actions ─► ledger ─► optimizer ─► gateway ─► arms ─► runner
            ▲            ▲           ▲          ▲
            └ evidence, provenance, roles, contagion, audit, metrics, channel (leaf helpers)
```
| module | responsibility | state | RNG |
|---|---|---|---|
| `config.py` | frozen `BaztConfig` (budget, TTLs, fusion, pacing, ablation switches) | – | – |
| `actions.py` | `ActionKind`, pure pricing (`b`, `u`, `net`, `density`), deterministic sort key | – | – |
| `ledger.py` | pooled / tiered budget, `charge` raises `BudgetViolation`, pacing `theta()` | spend | – |
| `evidence.py` | log-odds fusion, inspection verdict lifetimes | verdicts | – |
| `provenance.py` | message ids, parent links, taint propagation | `tau` | – |
| `roles.py` | ingress / relay / worker / privileged, articulation points | – | `bazt-roles` |
| `contagion.py` | memoised `H_i`, `H_edge` on the active, open graph (reuses `swarmlab.reach_H`) | per step | – |
| `optimizer.py` | two-phase deterministic plan | – | **none** |
| `gateway.py` | node states, blocked edges, inspection oracle, audit, hooks for `run_episode` | all runtime state | `bazt-inspect` (pre-drawn) |
| `audit.py` | append-only decision log, overspend recount | rows | – |
| `metrics.py` | §15 equations, legacy pricing, CRN-paired metrics | – | – |
| `channel.py` | rationale tag + edge-weight rule (price of filtering) | – | `bazt-sens` |
| `arms.py` | arm registry, budget equalisation, per-episode hooks/agent assembly | – | – |
| `runner.py` | CLI, grid, budget sweep, kickoff guard, safe output | – | – |

Every random draw comes from `swarmlab.stream_rng(<own namespace>, …)`, so adding BAZT cannot move a draw of the
simulator, and arms share their draws (strict CRN). The optimizer has no RNG at all.

## Control flow of one BAZT step (inside `run_episode`)
```text
consensus update   ── edge_open(j,i,t) gates senders; edge_weight_modifier scales each edge          (P4)
agents write       ── RationaleAgent appends `sens=hp|lp`
monitor            ── p_i, rolling score
StepView ───────────► gateway.decide(view, truth)                                                     (P5)
                       tau  ← lineage.advance        p' ← σ(a·roll − b + w_prov·tau) | verdict
                       phase 1: paid INSPECT ──► truth(i) only if conclusive
                       phase 2: SANITIZE / RESTRICT / BLOCK / ISOLATE by density ≥ θ(spent)
plan.isolate → quarantined      plan.sanitize + restricted(t) → trusted_sanitize (u_strip)            (P6)
gateway.on_deliver(t, {sender: stripped})
after step T:  gateway.extra_metrics(infected_series, world)  ← only place ground truth is used     (P7)
```

## Arms and what separates them
| arm | difference from `bazt` |
|---|---|
| `bazt_no_prov` | `use_provenance=False` (taint = 0, fusion = plain monitor) |
| `bazt_no_topo` | `H_i = 1` instead of graph reach (horizon rule kept) |
| `bazt_no_pacing` | flat `theta0` |
| `bazt_no_util` | `lam_eff = 0` (no utility penalty; EVPI of inspection is then 0) |
| `bazt_binary` | only `ISOLATE` candidates (no inspect, no soft actions) |
| `bazt_unlimited` | infinite budget (upper bound on what spending can buy) |
| `static_zta` | always-on role edges + schema-only channel, no budget |

## Extending
* New action: add to `ActionKind` (rank = tie-break), a `score_*` in `actions.py`, a cost field in `BaztConfig`, a pool
  in `BudgetLedger.pool_of`, and a branch in `BudgetedOptimizer.plan` / `EnforcementGateway._apply`.
* External monitor (Track 4): set `BaztConfig.probe_score_hook = f(node, t, StepView) -> Optional[float]`.
* Live LLM (after kickoff only): `python -m bazt.runner --mode llm --allow-live-llm [--evasion-adapter live_cot]`; the
  provider is `SWARMLAB_PROVIDER=openrouter|adaption`. Nothing in the code needs editing.

## Compatibility guarantees (all tested)
1. `run_episode` without hooks, with `EpisodeHooks()`, with a neutral weight modifier, or with a gateway factory but a
   legacy policy, is bit-identical to the original simulator (SHA-256 over 320 episodes).
2. `tests/test_core.py` is byte-identical to the pre-branch file (md5 pinned in a test).
3. `swarmlab.METRICS` is not extended; BAZT columns exist only in `results/bazt/`.
4. `results/per_seed_main.csv` and `results/per_seed_heldout.csv` are never written (md5 checked).
5. Scalability of the legacy path: the classic consensus loop is verbatim; A/B timing is within measurement noise (±2 %).
   BAZT cost per episode ≈ 11 ms (N=12), 38 ms (N=36), 136 ms (N=120): roughly linear in N.
