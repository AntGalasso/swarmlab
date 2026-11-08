# results/ — artifact map

What lives where. **Top-level legacy files are frozen**: `experiments.py` refuses to write them,
and their SHA-256 hashes are recorded in [`docs/LAUNCH_CHECKLIST_TOMORROW.md`](../docs/LAUNCH_CHECKLIST_TOMORROW.md)
(§ "Recorded legacy hashes"). Do not regenerate or edit them in place.

| path | what produced it | status |
|---|---|---|
| `per_seed_main.csv`, `per_seed_heldout.csv`, `summary_clean*.csv`, `summary_decoy*.csv` | `swarmlab.py --mode mock` legacy arms (main 1-30 / held-out 31-60) | **frozen** legacy baseline, never overwritten (enforced: writers refuse to clobber existing outputs; override one run with `SWARMLAB_ALLOW_OVERWRITE=1`) |
| `stats_report_main.json`, `stats_report_heldout.json` | `stats_check.py` on the legacy per-seed files | frozen |
| `tuned_params.json`, `tuning_summary.json` | `swarmlab.py --tune` (tune split 9001-9030) | frozen thresholds for deployable arms |
| `scaling_summary.csv`, `scaling_per_seed.csv` | `swarmlab.py --scaling-sweep` (N in 12/24/36) | frozen |
| `ladder_*` (breaking point, constraint evasion, stats, meta) | `env_ladder.py --all` sandbox ladder (calibration seeds 9401-9440) | frozen; **mock pipeline checks, not findings** |
| `figures/fig1-fig5*.pdf` | `make_figures.py` | publication figures |
| `bazt/` | `python -m bazt.runner` gateway outputs | sprint pipeline checks |
| `experiments/` | `experiments.py` pipeline: `mock_frontier/topology_scale/adaptive_*.jsonl`, `frozen_thresholds.json`, `manifest.json` | sprint runs (see `experiments/manifest.json` for provenance) |
| `experiments/paper_pack/` | `experiments.py paper-pack` (confirmatory stats + figures + tables) | **superseded pre-correction mock checks** — see `paper_pack/STATUS.md`; do not cite the old numbers |
| `ricardo/20261106/` | relay calibration + offline smoke receipts | completed; see `verification.json` |
| `ricardo/integrated_study_v6/` | frozen Gemma-26B campaign: 240 main + 120 held-out episodes, challenge gates, analysis, reproduction archives | **completed frozen evaluation**; SHA-256 of the archive in [`docs/RICARDO_EXECUTION_STATUS.md`](../docs/RICARDO_EXECUTION_STATUS.md). H2 unsupported, H3 mixed — the paper's primary live result |
| `integration/` | CI validation outputs (`validate_integration.py`, fake study) | regenerated per CI run |
| `logs/` | terminal logs from execution days | provenance only |

Labelling rule (README "Compliance guardrail"): **mock numbers are pipeline checks, never findings.**
Only the `ricardo/integrated_study_v6/` campaign and the live sweeps it links carry live-model evidence.
