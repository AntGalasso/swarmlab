# scripts/ops/

One-off status/monitoring scripts from the sprint execution days. They are **not** part of the
experiment pipeline; nothing imports them and CI does not run them (excluded from ruff via
`ruff.toml`).

| script | what it does |
|---|---|
| `check_pilot.py` | prints a summary of the latest Adaption pilot receipts under `results/` |
| `check_pilot_full.py` | same, over the full pilot output directory |

For the live recovery procedures, see `docs/TEAM_INTEGRATION.md` and the frozen-study drivers at
the repository root (`overnight_frozen_study.py`, `continue_frozen_study.py`).
