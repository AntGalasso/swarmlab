# swarmlab -- PRE-EXISTING SCAFFOLDING (declared for Apart Research compliance)

The original scaffolding was written **before** the Swarm Alignment Hackathon kickoff (Nov 6, 2026). Its original artifacts contain no live-model experimental result. **This includes the `bazt/` package, its tests and `docs/BAZT_SPEC_v2.md` / `docs/ARCHITECTURE_BAZT_FOR_REPO.md`
(branch `feat/bazt`, written Nov 4-5, 2026, before kickoff): every pre-sprint BAZT number comes from
`--mode mock` on tune seeds 9001-9012 and is a pipeline check, not a finding.** Organisers confirmed (Kamil, Apart) that pre-existing code is fine if marked as such in the
README and in the report. Everything measured with it during the sprint is sprint work.

Project: *When and Whom to Quarantine? Topology-aware risk control under alert-driven budget exhaustion in agent swarms.*

## Current team integration

The integration checkout includes `swarmlab` / `bazt`, the expanded `experiments.py` pipeline,
and Vittoria's `bazt_lab`. Use Python 3.11, initialize the pinned G-Safeguard submodule, and
run `uv sync --frozen` for the full project. See [docs/TEAM_INTEGRATION.md](docs/TEAM_INTEGRATION.md)
for source provenance, validation and the H2/H3 experiment gates. The pipelines keep separate
result semantics. CI is offline; inference is explicit and credentials are never committed.

## Completed evaluation and current extension

The frozen Gemma-26B campaign completed 240 main and 120 separately analyzed heldout episodes. H2 was unsupported and H3 was mixed; see [the verified status](docs/RICARDO_EXECUTION_STATUS.md) and [frozen results](results/ricardo/integrated_study_v6/analysis/). These are controlled simulator endpoints with live status messages, not general LLM safety measures.

A separate original-budget Early H2 and accessible-model H4 extension is running on new seeds. No new result is claimed before completion. The old `results/experiments/paper_pack` outputs are [superseded pre-correction mock checks](results/experiments/paper_pack/STATUS.md).

## Papers & companion materials

This repository is the code and evaluation-artefact companion for the team's three sprint papers
(authors: Antonio Galasso, Vittoria Lanzo, Ricardo Fitas, Kunal Sharma — with Apart Research).

1. **BAZT: Budgeted Agentic Zero-Trust Enforcement Against Prompt Contagion in LLM Agent Swarms** —
   gateway control plane, 2,400-episode simulator hypothesis suite (H1–H3), live qwen3.8-27b sweep and
   deepseek-v4-flash pilot. Code: `swarmlab.py`, `bazt/`, `stats_check.py`, `make_figures.py`.
   Artefacts: `results/stats_report_main.json`, `results/tuning_summary.json`, frozen episode records
   in `results/ricardo/integrated_study_v6/main/`, figures from `make_figures.py`. The pre-kickoff
   scaffolding snapshot is tagged `pre-kickoff-20261106T2000Z` (last commit before the kickoff guard
   2026-11-06T20:00:00Z).
2. **When Inspection Outlives Intervention: Budget-aware monitoring in multi-agent systems** — the
   `Stop` rule: skip inspection purchases once the intervention balance is below the smallest positive
   unit price. The inspection-budget contracts, conventional arms and the decision log behind it live
   in `bazt_lab/`, `budget_study.py` and `docs/EXPERIMENT_DECISIONS.md`.
3. **Excluded Agents Still Change the Score: A causal audit of multi-agent safety evaluation** — a
   causal audit of the public GAMMAF benchmark release (arXiv:2604.24477): editing one already-flagged
   agent's terminal vote can flip the recorded grade while the delivered answer is unchanged.
   Context: [docs/LITERATURE_CHECK.md](docs/LITERATURE_CHECK.md).

Reading order for reviewers: this README → [docs/ARCHITECTURE_MAP.md](docs/ARCHITECTURE_MAP.md) →
[results/README.md](results/README.md) (artifact map and freeze rules) → the papers.

## Repository layout
```
swarmlab.py  experiments.py  run_experiments.py      pipeline entry points (root: CI and tests import them by name)
stats_check.py  make_figures.py  env_ladder.py       confirmatory stats, publication figures, sandbox ladder
llm_client.py  live_cot_adapter.py                   live providers (kickoff-guarded)
budget_study.py  budget_study_analysis.py           cost study (CI-pinned) + its analysis library
overnight_frozen_study.py  continue_frozen_study.py  frozen-study drivers (digest-checked recovery procedure)
validate_integration.py                              CI validation entry point
bazt/            BAZT gateway package (13 modules + channel.py); python -m bazt.runner
bazt_lab/        modular-monolith experiment stack: features/<type>/<feature>/ with public.py + CONTRACT.md
configs/         frozen run configurations (sprint_preregistered.json, tier1_slice.json)
experiments/     profiles.toml for run_experiments.py
scripts/         check_static.py, run_adaption_pilot.py, swarmtraces_fetch.py, ops/ (archived one-offs), tests/
tests/           offline pytest suite (root-level)
docs/            every specification and status document; docs/archive/ holds superseded pre-integration drafts
vendor/          pinned G-Safeguard submodule (patch files only, no copied code)
results/         all artifacts — read results/README.md before touching anything (freeze rules per subtree)
```

## Compliance guardrail
* Before kickoff the ONLY live API call allowed is the READY smoke test (`python llm_client.py`).
* `--mode llm` needs `--allow-live-llm` AND is refused before the kickoff time (`SWARMLAB_KICKOFF_UTC`
  overrides the default 2026-11-06 20:00 UTC if the official start differs).
* Mock mode makes zero API calls. **Mock numbers are never findings.**
* Dual live payloads (`--payload canary_tag | consensus_drift`) exist only as templates in `swarmlab.py`;
  nothing containing `[REF-7]`, `[DOC-2]` or a decoy message is sent before kickoff.
* `python -m bazt.runner --mode llm` uses swarmlab's own kickoff guard: before kickoff it exits unconditionally (with or
  without `--allow-live-llm`); after kickoff it needs `--allow-live-llm` and works with `SWARMLAB_PROVIDER=openrouter|adaption`
  and `--evasion-adapter live_cot`, with no code edit. Its tests use a fake backend: zero live calls.

## Files
| file | role |
|---|---|
| `llm_client.py` | async OpenRouter client, SQLite WAL cache keyed by (model, messages, temperature, CRN salt), READY smoke test |
| `swarmlab.py` | simulator, 10 policy arms (incl. two-tier `risk_aware_tiered`), metrics, joint tuning, scaling sweep, LLM budget estimator, CLI |
| `stats_check.py` | pre-registered H1-H4 tests (paired Wilcoxon, Holm, bootstrap CI, Spearman) + strict Pareto / non-inferiority labels |
| `make_figures.py` | publication `.pdf` figures (safety-utility frontier, budget exhaustion dynamics, scaling curve) |
| `env_ladder.py` | Stage 1 fake-web corpus, Stage 2 difficulty ladder + breaking-point finder, Stage 3 hard constraints + `CoTPayloadAdapter` evasion sweep (mock-only, no live-LLM path) |
| `experiments.py` + `run_experiments.py` | expanded experiment pipeline (`llm-pilot` / `llm-tier1` / `mock-frontier` / `paper-pack`) + TOML-profile launcher (`experiments/profiles.toml`); refuses to write the legacy `results/per_seed_*.csv` |
| `budget_study.py`, `budget_study_analysis.py`, `overnight_frozen_study.py`, `continue_frozen_study.py` | budget cost study (run by CI) and the digest-checked frozen-study drivers behind the completed overnight campaigns |
| `bazt/` | Budgeted Adaptive Zero-Trust gateway (13 modules + `channel.py`): `python -m bazt.runner`; writes only to `results/bazt/` |
| `bazt_lab/` | modular-monolith experiment stack (mock engine, G-Safeguard llm_harness + patch set, analysis, judge, tracking, gnn); see `docs/ARCHITECTURE_EXPERIMENTS.md` |
| `configs/` | frozen run configurations: `sprint_preregistered.json`, `tier1_slice.json` |
| `scripts/` | `check_static.py` (ruff + compileall + import gate), `run_adaption_pilot.py`, `swarmtraces_fetch.py`, `ops/` archived one-off status scripts |
| `docs/` | all specifications, experiment plans, metrics equations, integration and execution status; `docs/archive/` = superseded pre-integration drafts |
| `results/` | artifact map and freeze rules: `results/README.md` |
| `docs/BAZT_SPEC_v2.md`, `docs/ARCHITECTURE_BAZT_FOR_REPO.md` | BAZT specification with resolved decisions; module map and extension guide |
| `docs/METRICS_SPEC.md` | exact equations for every reported metric, monitor, risk score, tuning objective and label (§1-§13), the ladder/sandbox metrics (§14) and the BAZT gateway metrics (§15) |
| `docs/KUNAL_SANDBOX_GUIDE.md` | developer guide to the three stages: how to plug a live CoT payload adapter into Stage 3 after kickoff |
| `docs/ARCHITECTURE_MAP.md` | module map, seed splits, artifact list, per-teammate ownership |
| `scripts/swarmtraces_fetch.py` | download + aggregate-only profile of the public Swarm Traces dataset |
| `tests/test_core.py` | 46 offline tests (determinism, CRN prefix sharing, caps, guards, cache, decoy realism, arm divergence, metric formulas, Pareto labels, per-step tier-1 budgets, payloads) |
| `tests/test_env_ladder.py` | 27 offline tests (corpus schema/determinism, ladder monotonicity + pinned breaking point, routing masks, channel filters, adapter contract, hook compatibility, LLM refusal) |
| `tests/test_bazt_{core,gateway,integration}.py` | 92 offline tests: pricing, ledger, lineage, optimizer, gateway state machine, §15 equations, legacy bit-identity, ground-truth firewall, budgets, filtering cost, kickoff guard, dependency direction |

## Quickstart
```bash
pip install -r requirements.txt && cp .env.example .env     # add OPENROUTER_API_KEY
pytest -q                                                    # offline, no key needed
python llm_client.py                                         # only live call allowed before kickoff

python swarmlab.py --mode mock --tune                        # full tune split 9001-9030, joint objective J
python swarmlab.py --mode mock --split main    --params results/tuned_params.json
python swarmlab.py --mode mock --split heldout --params results/tuned_params.json
python swarmlab.py --mode mock --scaling-sweep --seeds 20    # N in {12,24,36} on small_world
python stats_check.py --file results/per_seed_main.csv
python stats_check.py --file results/per_seed_heldout.csv
python make_figures.py                                       # fig1..fig3

python env_ladder.py --all                                   # corpus + ladder + Stage-3 sweep (~20 s, mock)
python env_ladder.py --build-corpus                          # only data/fake_web_corpus.jsonl
python swarmlab.py --mode mock --entry-source fake_web_jsonl --seeds 2 --topologies chain        --policies no_defense risk_aware_tiered --out results/entry_demo
python stats_check.py --ladder results/ladder_breaking_point.csv                       --evasion results/ladder_constraint_evasion.csv --out results/ladder_stats_report.json
python make_figures.py                                       # fig1..fig5 (fig4/fig5 need the ladder outputs)
python swarmlab.py --budget --seeds 30 --price-in <usd/Mtok> --price-out <usd/Mtok>   # you supply real prices
python scripts/swarmtraces_fetch.py download && python scripts/swarmtraces_fetch.py profile

python -m bazt.runner --mode mock --split tune --seeds 12     # BAZT smoke check on tune seeds -> results/bazt/smoke_tune_check.csv
python -m bazt.runner --mode mock --split tune --budget-sweep # cost-matched sweep of the pooled budget B
```
Seed splits: **tune** 9001-9030 (always all 30 for tuning), **main** 1-30, **heldout** 31-60, **ladder calibration**
9401-9440 (freezes the ladder's success bar, never graded), **ladder/scaling** 9001-9020. `--tune` refuses other splits.
All deployable arms are tuned on the same tune seeds, optimising
`J = AUC_ever + alpha * (MAE / MAE_baseline)` (`--alpha`, default 0.25) so a candidate cannot buy containment
by wrecking consensus accuracy. Artifacts: `results/tuning_summary.json`, `results/tuned_params.json`.

Outputs never overwrite each other: `per_seed_main.csv` / `per_seed_heldout.csv`,
`episodes_main.jsonl` / `episodes_heldout.jsonl`, `summary_clean.csv` + `summary_decoy.csv` (main) and
`summary_*_heldout.csv` (held-out).

## Two-tier protocol (`risk_aware_tiered`)
A graduated response separates *contagion neutralisation* from *information destruction*:

* **Tier 1 (soft, `B_san = 6` actions/episode, PER STEP)** -- `trusted_sanitize()`: strict numeric-schema
  extraction keeps `est=<number>` (so the node still informs consensus) and drops every tag/free-text token
  (so `[REF-7]` stops propagating for that step), with success probability `p_strip = 0.90` in mock mode.
  A node that stays risky must be re-selected and spends another action; nothing is silenced permanently.
* **Tier 2 (hard, `K = 3`/episode)** -- `R >= tau_quar`: permanent outgoing-edge isolation.
* A tier-2 candidate that no longer fits in `K` **degrades to tier 1** instead of being dropped.
* H3 is `supported` when `risk_aware*` reaches a strict Pareto or bounded-utility win on >=50% of the
  pre-registered comparisons with zero significant MAE losses (`dominance_rate` is reported per split).

## Sprint-day LLM run (after kickoff)
```bash
python swarmlab.py --mode llm --allow-live-llm --model qwen/qwen-2.5-7b-instruct --split main \
    --payload canary_tag --topologies star_hub star_leaf small_world --params results/tuned_params.json --tag qwen
python swarmlab.py --mode llm --allow-live-llm --model openai/gpt-4o-mini --split main \
    --payload consensus_drift ... --tag gpt4omini   # H4
python stats_check.py --file results/per_seed_main_qwen.csv --h4 results/per_seed_qwen.csv results/per_seed_gpt4omini.csv
python make_figures.py --file results/per_seed_main_qwen.csv --episodes results/episodes_main_qwen.jsonl
```

## Design choices to state in the paper
* Monitor score is computed from the OBSERVED (pre-filter) output (canary / decoy tag) plus CRN noise;
  evidence is a zero-padded rolling mean over m steps, so a freshly infected agent's first-step evidence
  equals a single decoy burst's.
* Decoy realism: `[DOC-2]` headers arrive in a short (1-2 step) early work-cycle burst and then stop, so a
  decoy is a *transient* spike (`mu_dec = 2.8 > mu_inf/2`) while contagion is persistent. Three such benign
  alerts exhaust an early `K = 3` budget and let patient zero survive.
* `fixed_*` arms are time-boxed snapshot audits (one threshold crossing fires them); `score_only` has no
  window and needs `confirm_steps = m` consecutive crossings. These are different decision paths, so their
  trajectories diverge under decoy pressure (tested).
* Quarantine acts before delivery of the current step's message. `edge_isolation`: frozen estimate stays in
  the readout; `node_removal`: dropped from readout.
* Numeric consensus (Metropolis weights) is simulator-computed; infected agents report a shifted number
  (`bias`). In LLM mode the model writes the status line, i.e. the contagion channel.
* Monitor is parametric, not a learned model. Decoy intensity (`--mu-dec`) is a controlled parameter.
* Tier-1 sanitation preserves the numeric estimate, so its measured utility cost is only re-quantization +
  the `B_san` budget; the loss of qualitative coordination is not measured by this task (see
  `docs/METRICS_SPEC.md` §13).

## Sandbox ladder (`env_ladder.py`, mock only)
A third question, upstream of the quarantine comparison: **at what difficulty does the swarm stop agreeing at
all?** Stage 1 seeds the episode from a deterministic 36-document fake-web corpus (28 clean, 4 benign
`[DOC-2]` decoys, 4 benign entry payloads wrapped in `story_context` / `base64_stub` /
`markdown_metadata`). Stage 2 tightens five orthogonal knobs over levels 1-8 and freezes
`target_mae_threshold` on calibration seeds 9401-9440, then reports
`breaking_level = min{l : pass_rate_l < 0.50}` and `critical_operating_level = max(1, breaking_level - 1)`.
Stage 3 holds the ladder there and adds `HardConstraintSpec` (movement: `unrestricted`, `directed_dag`,
`chokepoint_hub`, `ring_clockwise`; channel: `free_note`, `banned_brackets`, `length_clamped_32`,
`structured_kv_only`) plus a `CoTPayloadAdapter` seam that reports **payload delivery** separately from what
the m-step monitor can see.

Result to quote (mock, level 4): a naive `[REF-7]` payload is destroyed by `banned_brackets` (delivery 0.00)
while a compressed stand-in is delivered 100 % of the time and alarms the monitor on ~4 % of node-steps - the
lexical blacklist is beaten. `structured_kv_only` reduces delivery to 0.00 for both forms, because it removes
the free-text channel instead of filtering strings: **architectural mediation beats lexical blacklisting**.
`env_ladder.py` has no live-LLM code path (`--mode llm` is refused); the live adapter is a subclass, see
`docs/KUNAL_SANDBOX_GUIDE.md`. All numbers are pipeline checks, not findings.

## BAZT gateway (`bazt/`, branch `feat/bazt`)
A budget-constrained enforcement layer between agents: it pays to *inspect* a node, *sanitize* its channel, *restrict* it,
*block* an edge or *isolate* it, ranked by `density = (benefit - lam*(1-p')*utility_cost) / cost` against a threshold that
rises as the budget drains. Evidence fuses the monitor with lineage taint; ground truth is reachable only through a paid
inspection. Arms: `bazt` + six single-mechanism ablations, `static_zta`, and the legacy arms priced in the same cost units
(`--equalize-budget`). It attaches through `EpisodeHooks.gateway_factory`, so every legacy arm is bit-identical (SHA-256 test).

* Filtering is not free: a calibration rationale (`sens=hp|lp`) rides on every message and a stripped message keeps only
  `stripped_edge_weight = 0.35` of its consensus weight, so over-sanitising benign nodes raises `MAE` (docs/METRICS_SPEC.md §15.6).
* Seeds: tune `BaztConfig` on **9001-9030 only**; `--split main|heldout` needs `--confirm-frozen-config`.
* Outputs live in `results/bazt/`; `results/per_seed_main.csv` / `per_seed_heldout.csv` are never written (artifact map: `results/README.md`).
* Known limit of the untuned defaults (docs/BAZT_SPEC_v2.md §10): the budget is spent almost entirely at step 1.
