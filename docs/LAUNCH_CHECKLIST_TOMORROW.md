# docs/LAUNCH_CHECKLIST_TOMORROW.md

> **Copy-paste PowerShell for tomorrow morning (after `SWARMLAB_KICKOFF_UTC = 2026-11-06 20:00 UTC`).**
> Everything below was already exercised offline today with `--platform mock` and a deterministic
> `FakeAsyncLLMBackend`: **zero live LLM calls happened before kickoff.**

---

## 0. Open the right shell

```powershell
cd "C:\Users\Antonio\Desktop\dev tools\hacklaton\swign alinment\swarmlab_v3"
. .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
```

## 1. Pre-flight (safe at any time, no live calls)

```powershell
python -m pytest -q
python experiments.py calibrate --config configs/sprint_preregistered.json --platform mock --split tune
```

Expected: `0 failed`, and `results/experiments/frozen_thresholds.json` written with a `config_sha256`.

## 2. Confirm the provider and keys (values stay in `.env`, never in the shell history)

```powershell
$env:SWARMLAB_PROVIDER = "openrouter"      # or "adaption"
# keys live in .env (OPENROUTER_API_KEY / OPENAI_API_KEY / ADAPTION_API_KEY) -- do NOT type them here
```

Optional provider switch (no code edit needed either way):

```powershell
$env:SWARMLAB_PROVIDER = "adaption"        # requires ADAPTION_API_KEY and `pip install adaption`
```

## 3. The ONE live call allowed before kickoff: the READY smoke test

```powershell
python llm_client.py
```

Must print `PASS` for the primary and H4 arms. If it fails, stop here — do not start the sprint runs.

## 3b. (Optional) rehearse the whole LLM pipeline offline, right now

```powershell
python experiments.py llm-pilot --config configs/sprint_preregistered.json --platform llm `
  --fake-backend --smoke --resume --out results/experiments/_offline_llm_smoke
```

`--fake-backend` uses the deterministic `FakeAsyncLLMBackend` (no network, no key, no cost), so this is legal
before kickoff and proves the judge + 8-conversation persistence path end-to-end. Delete the `_offline_llm_smoke`
folder afterwards so simulated rows never reach the paper pack.

## 4. Confirm the kickoff gate is armed (still refused if you are early)

```powershell
python experiments.py llm-pilot --config configs/sprint_preregistered.json --platform llm --allow-live-llm --resume
```

Before 20:00 UTC this exits with `Refusing: before kickoff (...)`. That is the guard working.
**After kickoff** the same command starts the pilot.

## 5. Post-kickoff launch order

```powershell
# 5a. pilot: 2 seeds x 4 conversations, live
python experiments.py llm-pilot --config configs/sprint_preregistered.json --platform llm --allow-live-llm --resume

# 5b. Tier-1: 12 seeds x 8 conversations, 3 scenarios x 2 topologies,
#     uncapped A/B/C/BAZT + capped C and BAZT at (2%,2%), (10%,10%), (50%,50%)
python experiments.py llm-tier1 --config configs/sprint_preregistered.json --platform llm --allow-live-llm --resume

# 5c. mock stages (idempotent; --resume makes a re-run free)
python experiments.py mock-frontier        --config configs/sprint_preregistered.json --split main --resume
python experiments.py mock-topology-scale  --config configs/sprint_preregistered.json --split main --resume
python experiments.py mock-adaptive        --config configs/sprint_preregistered.json --split main --resume

# 5d. paper pack: confirmatory statistics + 4 figures + 2 tables
python experiments.py paper-pack --config configs/sprint_preregistered.json
```

## 6. If anything crashes at 3 a.m.

Every episode is appended as ONE line with `O_APPEND` + `fsync`, and `--resume` skips any
`(platform, stage, split, scenario, topology, N, arm, audit_pct, int_pct, seed)` tuple that is already on disk.
A half-written final line is ignored and that episode is simply re-run.

```powershell
# just re-run the interrupted command with --resume
python experiments.py llm-tier1 --config configs/sprint_preregistered.json --platform llm --allow-live-llm --resume
```

## 7. Held-out split (only after the config is frozen and main is analysed)

```powershell
python experiments.py mock-frontier --config configs/sprint_preregistered.json --split heldout `
  --confirm-frozen-analysis --resume
```

Without `--confirm-frozen-analysis` this refuses to run. That is intentional.

## 8. Verify the legacy artifacts were never touched

```powershell
Get-FileHash results\per_seed_main.csv    -Algorithm SHA256
Get-FileHash results\per_seed_heldout.csv -Algorithm SHA256
```

Compare against the values recorded at the bottom of this file. Any difference is a release blocker.

## 9. Publish the pack

```powershell
Get-ChildItem results\experiments\paper_pack
```

Expected: `fig_exp1_capped_gains.pdf`, `fig_exp2_dual_budget_frontier.pdf`,
`fig_exp3_topology_scale_ablations.pdf`, `fig_exp4_adaptive_and_exhaustion.pdf`,
`table1_primary_results.md`, `table1_primary_results.tex`,
`table2_ablations_and_scale.md`, `table2_ablations_and_scale.tex`,
`confirmatory_stats.json`.

---

## Recorded legacy hashes (fill in before you start; they must not change)

| file | SHA256 before launch | SHA256 after the mock dry-run |
|---|---|---|
| `results/per_seed_main.csv` | `85b39908000b159e0f42191f0d25f46afc34171f41ef7ce1c3b93cb75a15e1b0` | **identical** |
| `results/per_seed_heldout.csv` | `0c210c1f519b37f50c171874286a01d1012f62cd4d6d8b37a93b24eb71ed6c7f` | **identical** |

## Compliance reminders

* Before kickoff: only `python llm_client.py` may make a live call. `--platform llm` is refused outright.
* Mock numbers are **pipeline checks, never findings** — label every mock table as such.
* All new artifacts live under `results/experiments/`; `experiments.py` refuses to write the legacy
  `per_seed_main.csv` / `per_seed_heldout.csv` at all.
* `--split heldout` needs `--confirm-frozen-analysis`, and calibration only ever runs on tune seeds.
