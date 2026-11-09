# António: independent N12/N100 CPU diagnostic

This task is reserved for António. Our parallel run42386290416 is **clean calibration only**; this package does not start a remote job. It runs no LLM/API calls and requires no API key. It does not run legacy experiments.py or global run-all.

Keep the PR checkout separate from any directory where your existing work is running:

```sh
# In your existing team repository, fetch this PR branch without switching your current checkout.
git fetch origin handoff/antonio-n100-20261107
git worktree add ../swarmlab-antonio-handoff FETCH_HEAD
cd ../swarmlab-antonio-handoff
```

Then the one command is:

```sh
uv run python bazt_handoff_antonio/run_n100.py --workers 8
```

For inspection only, append --dry-run. It validates the frozen profile and prints the two commands; it does not fetch, create a runtime, calibrate or evaluate.

The launcher creates/reuses a **second detached runtime worktree pinned exactly to a88f146675134133b85ae7eaf16d38048096c7c6**. Its source is checked clean before running. This preserves the baseline used by the frozen main study while the PR itself can be based on current team main32e92f45. No source revert or edit is applied to your new gateway, calibration, live harness or other ongoing work. If the old commit is missing locally, only that commit is fetched from origin. Dependencies are installed/reused by uv from the pinned runtime's frozen lockfile.

## Scope and ownership

- Populations12and100; BA, star_hub and small_world; five parameterized attack-family labels plus their clean conditions.
- C, core BAZT and BAZT_upstream at10%/10%anduncapped;30independent evaluation seeds1–30.
- One keyed protected sink per seed at **both** population sizes. This is a separate, exploratory one-target scalability diagnostic, not the all-sink N12frontier or confirmation of the full H-topology claim.
- Clean calibration seeds9001–9150, closed loop with up to40candidate passes. Heldout31–60, adaptive, GNN and live tasks remain unused.
- Same frozen scientific parameters as the previously specified N100diagnostic; the batch size32is a memory/performance setting. Workers1–8are allowed; each torch worker uses one thread.

"Independent task" means separate ownership/execution. N12seeds1–30reuse the main frontier's underlying world set; they are not a new independent N12replication and must not be pooled as n60. N100draws different graphs/readings through keys containing N. Pair arms by seed within each N; equal seed numbers do not make N12andN100the same episode.

Default outputs go only to this checkout's results/antonio_n100_20261107/. The launcher refuses a nonempty output folder without a matching handoff manifest and refuses the existing results/experiments/live folder. A process lock prevents duplicate local writers. To choose another fresh folder, pass --results-root with its path. Never point it at your existing live or frozen-threshold folders.

## Steps, time and restart

The command runs exactly mock_calibration, checks the required18evaluation calibration cells, then runs exactly mock_epidemiology. It refuses an achieved calibration rate above2.2%or missing calibration. Below-band cells are kept and flagged; this does not mean exact equal FPR. No parameter search uses attacked results.

Calibration can be substantial at N100and is serial even with eight workers. Its nominal grid has3600clean episodes before extra closed-loop passes; do not assume it will take only minutes. The evaluation has6480policy-condition records. Installation, calibration passes, hardware and I/O determine elapsed time. Both blocks save checkpoints. If interrupted, rerun the **same command with the same output path**; completed cells/units are reused. The default does its own clean calibration and needs no access to our private execution repository or artifacts.

Outputs include:

- ANTONIO_N100_MANIFEST.json: source pin, exact settings digest and protocol;
- CALIBRATION_GATE.json and per-cell thresholds/diagnostics;
- antonio_n100_diagnostic/mock_calibration/ and mock_epidemiology/ manifests, settings and committed units;
- antonio_n100_diagnostic/COMPLETENESS.json: exact6480units,216cells,30seeds each and zero budget violations;
- antonio_n100_diagnostic/n100_summary.csv: descriptive ASR, clean/attacked accuracy, excess error against the paired clean reference, targeted/sink harm, propagation, spend and FPR, with valid denominators.

Send the manifest, calibration gate, completeness report and summary CSV back to the team; retain raw committed units for reproduction. No results, keys or private manuscript are included in this PR.

## Interpretation already fixed

ASR includes ordinary final consensus error. Report the paired clean floor and excess task error; the short star horizon is much shorter than its mixing time, especially at N100. Keep eps2rather than tuning it after attacks. The rate limit99is inactive at N≤100. The three insider labels share the declared mock bias/detectability assumptions. Both BAZTs have free monitor evidence while C uses paid audits, so this compares complete systems under that explicit information asymmetry. Statistical replication is by seed, never by individual agents or the6480records.

Offline checks:

```sh
uv run python -m unittest discover -s bazt_handoff_antonio -p test_handoff.py -v
uv run python bazt_handoff_antonio/run_n100.py --workers 8 --dry-run
```
