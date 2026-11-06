# Integrated team execution

The integration branch combines the lightweight simulator/gateway, Antonio's
expanded experiment pipeline and Vittoria's `bazt_lab` in one checkout. The two
experimental pipelines retain their own configurations, observation boundaries
and result schemas; merging code does not pool their measurements.

Sources are recorded in `INTEGRATION_SOURCE.json`. The original scaffold and
pre-sprint mock artifacts remain pre-existing work. Transport corrections,
integration checks and post-kickoff measurements are recorded separately.

## Installation and checks

Use Python 3.11 and `uv sync --frozen` for the full project. The pinned CPU torch
build is sufficient for offline validation; no pretrained model download is
needed. Initialize the pinned `vendor/G-safeguard` submodule for the harness and
upstream parity tests. The original `requirements.txt` supports the lightweight
path only.

```sh
git submodule update --init
uv sync --frozen
uv run python validate_integration.py
uv run python scripts/check_static.py
uv run python budget_study.py --phase fake --out results/integration/local_fake
```

The grouped validator runs every root, lab and script test in a fresh process,
with CPU thread counts bounded. It retains individual logs and JUnit summaries.
This avoids retaining the combined ML compiler/import caches across the suite.
The harness parity loader does not write bytecode into the pinned submodule.
The isolated-audit reference digest was independently reproduced on the original
PR #6 and the merged code; the complete numeric rows are identical (see
`docs/INTEGRATION_REGRESSION_PROOF.json`).

Push and PR verification are offline. Paid inference requires an explicit live
command or an explicit manual dispatch; no repository secret is required by CI.

## Live bridge

The real `LLMBackend` is bridged to the expanded conversation harness through
`bazt/live_backend_adapter.py`. Sync agent calls and async judge calls use the
same cached-client event loop. The present transports expose no first-token
log probabilities, so live judging uses an explicitly labelled structured score.
Malformed live judge replies raise instead of becoming synthetic hash scores.
The live factory requires a different judge model and closes both clients.

This bridge is independent of the `bazt_lab` transport and judge implementation.
The primary lightweight study below uses a parametric simulator monitor and a
paid simulator-side inspection oracle. It does not claim to validate an LLM
semantic judge or a trained GNN.

## H2/H3 study gates

`budget_study.py` keeps proxy tuning, attack-free controls, unprotected challenge
validation and evaluation separate. The harmless relay note is instructed only
to patient zero and reaches other agents through their inboxes. Healthy agents
are not directly instructed to adopt it when exposed.

1. `--phase tune-mock`: all 30 tune seeds, identical joint objective for the
   deployable comparators, and the same total intervention cost cap.
2. `--phase challenge`: five new tune seeds and two topologies, N=12 and T=12,
   with genuine attack-free controls. No incomplete episode is accepted.
3. `--phase main`: freeze parameters/source before 12 independent evaluation
   seeds. Score-only and centrality-only baselines, BAZT and its no-pacing
   ablation share a 12-unit cap; no defence is the untreated reference.
4. `--phase heldout --confirm-frozen-analysis`: six reserved seeds, after main.

The first integrated challenge attempt at 512 output tokens stopped on an
incomplete response before accepting any episode. Version 2 also stopped on an
incomplete response at 2048 tokens. Version 3 makes at most one larger-budget
retry (2048, then 8192 tokens) and rejects incomplete output. These transport
changes were made before main evaluation; the original failed receipts are
retained. Resuming an interrupted phase reuses completed cached requests and
keeps an additional usage receipt for every attempt. API credit calculations are
separate from enforcement units and
are not verified account invoices.

The statistical unit is the seed, with paired contrasts averaged across
topologies. H2 concerns the decoy penalty of the score-only reference; H3 concerns
BAZT's safety/utility comparison. Predefined paired tests and uncertainty are
reported even for negative or mixed results. No mock number is an empirical
LLM finding.
