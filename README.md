# swarmlab -- PRE-EXISTING SCAFFOLDING (declared for Apart Research compliance)

Written **before** the Swarm Alignment Hackathon kickoff (Nov 6, 2026). It contains no experimental
result. Organisers confirmed (Kamil, Apart) that pre-existing code is fine if marked as such in the
README and in the report. Everything measured with it during the sprint is sprint work.

Project: *When and Whom to Quarantine? Topology-aware risk control under alert-driven budget exhaustion in agent swarms.*

## Compliance guardrail
* Before kickoff the ONLY live API call allowed is the READY smoke test (`python llm_client.py`).
* `--mode llm` needs `--allow-live-llm` AND is refused before the kickoff time (`SWARMLAB_KICKOFF_UTC`
  overrides the default 2026-11-06 20:00 UTC if the official start differs).
* Mock mode makes zero API calls. **Mock numbers are never findings.**

## Files
| file | role |
|---|---|
| `llm_client.py` | async OpenRouter client, SQLite WAL cache keyed by (model, messages, temperature, CRN salt), READY smoke test |
| `swarmlab.py` | simulator, 9 policy arms, metrics, tuning, LLM budget estimator, CLI |
| `stats_check.py` | pre-registered H1-H4 tests (paired Wilcoxon, Holm, bootstrap CI, Spearman) |
| `swarmtraces_fetch.py` | download + aggregate-only profile of the public Swarm Traces dataset |
| `tests/test_core.py` | 21 offline tests (determinism, CRN prefix sharing, cap, guards, cache) |

## Quickstart
```bash
pip install -r requirements.txt && cp .env.example .env     # add OPENROUTER_API_KEY
pytest -q                                                    # offline, no key needed
python llm_client.py                                         # only live call allowed before kickoff
python swarmlab.py --mode mock --tune --seeds 20             # tunes thresholds on seeds 9001-9030 only
python swarmlab.py --mode mock --split main --params results/tuned_params.json
python stats_check.py --file results/per_seed.csv
python swarmlab.py --budget --seeds 30 --price-in <usd/Mtok> --price-out <usd/Mtok>   # you supply real prices
python swarmtraces_fetch.py download && python swarmtraces_fetch.py profile
```
Seed splits: **tune** 9001-9030 (thresholds), **main** 1-30, **heldout** 31-60. `--tune` refuses other splits.
All deployable arms are tuned (a score-only baseline left at an arbitrary threshold would be a strawman).

## Sprint-day LLM run (after kickoff)
```bash
python swarmlab.py --mode llm --allow-live-llm --model qwen/qwen-2.5-7b-instruct --split main \
    --topologies star_hub star_leaf small_world --params results/tuned_params.json --tag _qwen
python swarmlab.py --mode llm --allow-live-llm --model openai/gpt-4o-mini --split main ... --tag _gpt4omini   # H4
python stats_check.py --file results/per_seed_qwen.csv --h4 results/per_seed_qwen.csv results/per_seed_gpt4omini.csv
```

## Design choices to state in the paper
* Monitor score is computed from the OBSERVED output (canary / decoy tag) plus CRN noise; evidence is a
  zero-padded rolling mean over m steps, so a freshly infected agent initially looks like a decoy.
* Quarantine acts before delivery of the current step's message. `edge_isolation`: frozen estimate stays in
  the readout; `node_removal`: dropped from readout.
* Numeric consensus (Metropolis weights) is simulator-computed; infected agents report a shifted number
  (`bias`). In LLM mode the model writes the status line, i.e. the contagion channel.
* Monitor is parametric, not a learned model. Decoy intensity (`--mu-dec`) is a controlled parameter.
