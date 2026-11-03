# ARCHITECTURE_MAP.md — one page: modules, seeds, artifacts, ownership

Branch: `feat/kunal-env-ladder`. Project: *When and Whom to Quarantine? Topology-aware risk control under
alert-driven budget exhaustion in agent swarms.* Everything below runs offline in `--mode mock`.

## Data flow

```text
                    ┌──────────────────────────── Stage 1 (entry) ────────────────────────────┐
  data/fake_web_corpus.jsonl (36 docs)  ──entry agents──►  swarmlab.run_episode / run_grid       │
  28 clean | 4 benign [DOC-2] decoys | 4 benign entry payloads    (hooks=EpisodeHooks)           │
                    └──────────────────────────────────────────────────────────────────────────┘
                                                     │
                    ┌──────────────── Stage 2 (reusable sandbox difficulty) ────────────────────┐
  env_ladder.DifficultyLevelSpec (levels 1-8) ──► BreakingPointFinder ──► breaking_level,          │
     5 knobs: topology, N, T, sensor_sigma/obs_noise, bandwidth                    critical_level  │
                    └──────────────────────────────────────────────────────────────────────────┘
                                                     │  (constraints held fixed here)
                    ┌──────────────── Stage 3 (attack / defence at that point) ────────────────┐
  HardConstraintSpec (movement x channel) + CoTPayloadAdapter ──► delivery vs detection vs MAE    │
   policies: no_defense, fixed_early, score_only, risk_aware, risk_aware_tiered                    │
                    └──────────────────────────────────────────────────────────────────────────┘
                                                     ▼
                 results/*.csv|json ──► stats_check.py (paired tests) ──► make_figures.py (fig1..fig5)
```

## Modules

Owners for everyone except Kunal are the working assumption from the branch/task split — confirm them at
kickoff and edit this table rather than guess later.

| file | role | owner |
|---|---|---|
| `swarmlab.py` | simulator: N-agent swarm, contagion, m-step monitor, 10 policy arms, `risk_aware_tiered`, joint tuning `J`, scaling sweep, CLI. **Pre-existing scaffolding** — declare it as such. | Antonio |
| `env_ladder.py` | Stage 1 corpus, Stage 2 ladder + `BreakingPointFinder`, Stage 3 `HardConstraintSpec` / `CoTPayloadAdapter` / evasion sweep. Mock-only, no live-LLM path. | Kunal |
| `KUNAL_SANDBOX_GUIDE.md` | how to plug a live CoT adapter into Stage 3 after kickoff | Kunal |
| `llm_client.py` | async OpenRouter client, SQLite WAL cache keyed by `SHA256(model, messages, round(temp,4), crn_salt)`, READY smoke test, pre-kickoff guard | Vittoria |
| `stats_check.py` | pre-registered H1–H4 (paired Wilcoxon, Holm, bootstrap CI, Spearman), strict-Pareto labels, `--ladder` / `--evasion` reports | Ricardo |
| `make_figures.py` | `fig1` frontier, `fig2` budget exhaustion, `fig3` scaling, `fig4` ladder, `fig5` constraint evasion | Ricardo |
| `swarmtraces_fetch.py` | download + aggregate-only profile of the public Swarm Traces dataset (no raw prompt dumps) | Anirudh |
| `METRICS_SPEC.md` | exact equations §1–§13 (main pipeline) and §14 (ladder/sandbox) | Antonio + Ricardo |
| `tests/test_core.py` | 46 pre-existing offline tests — the compatibility contract, do not edit | Antonio |
| `tests/test_env_ladder.py` | 27 tests for corpus, ladder, masks, filters, adapter, guards | Kunal |
| `README.md` | compliance guardrail, file table, quickstart | Antonio |

## Seed splits (never overlap — this is the pre-registration)

| split | seeds | used for | written to |
|---|---|---|---|
| main | 1–30 | primary evaluation, H1–H3 | `results/per_seed_main.csv` |
| held-out | 31–60 | confirmation only, run **after** the analysis is frozen | `results/per_seed_heldout.csv` |
| tune | 9001–9030 | policy tuning (`J`) and the ladder's `pass_rate` (9001–9020) | `results/tuned_params.json`, `results/tuning_summary.json` |
| ladder-calibration | 9401–9440 | **freezing** `τ_l` only; never graded | `env_ladder.CALIB_SEEDS` |
| scaling | 9001–9020 | `N ∈ {12,24,36}` on `small_world` | `results/scaling_*.csv` |

`tests/test_env_ladder.py::test_calibration_seeds_are_disjoint_from_every_graded_split` enforces the
disjointness. Changing a split boundary invalidates every number in the results folder.

## Artifacts

| artifact | content |
|---|---|
| `results/per_seed_{main,heldout}.csv` | one row per `(topology, regime, policy, seed)`: AUC, MAE, quarantines, decoy waste, `dRel` |
| `results/summary_{clean,decoy}[_heldout].csv` | descriptive tables (never mixed across regimes) |
| `results/stats_report_{main,heldout}.json` | H1–H4 with `p_raw` **and** `p_holm`, Pareto labels |
| `results/tuned_params.json`, `results/tuning_summary.json` | joint-objective tuning, full candidate table |
| `results/scaling_per_seed.csv`, `results/scaling_summary.csv` | scaling sweep, both budget rules |
| `results/ladder_breaking_point.csv` | one row per `(level, seed)`: `task_mae`, `passed`, `τ_l`, `message_drop_rate` |
| `results/ladder_summary.json` | `breaking_level`, `critical_operating_level`, per-level `pass_rate` + Wilson CI |
| `results/ladder_constraint_evasion.csv` | one row per `(level, movement, channel, payload_form, policy, seed)` |
| `results/ladder_constraint_summary.csv` | per-cell delivery / detection / gap / `stealth_goal_joint` / `auc_ever` |
| `results/ladder_constraint_meta.json` | sweep metadata + the two success rules + compliance note |
| `results/ladder_stats_report.json` | `stats_check.py --ladder --evasion` (Wilson CI, paired Holm) |
| `results/figures/fig1..fig5*.pdf` | publication figures |
| `data/fake_web_corpus.jsonl` | deterministic Stage-1 corpus (yaml/jsonl under `data/` is gitignored) |

## Boundaries that must not move

1. `swarmlab.run_episode(policy, cfg, agent, decoys)` with no `hooks` is **bit-identical** to the pre-existing
   simulator — all 46 legacy tests are the proof. New behaviour arrives only through `EpisodeHooks`.
2. `env_ladder.py` never imports `llm_client` and never builds a backend: `--mode llm` is refused outright.
   Live work belongs in a subclass of `CoTPayloadAdapter`, after `SWARMLAB_KICKOFF_UTC`.
3. Split files are append-free: a rerun overwrites only its own file. `main`/`heldout` are never touched by
   tuning, calibration or the ladder.
4. Every random draw comes from `swarmlab.stream_rng(namespace, key, seed)`; no global RNG anywhere.
