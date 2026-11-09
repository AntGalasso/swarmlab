#!/usr/bin/env python3
"""Report achieved calibration rates; never silently equate requested and achieved alpha."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("thresholds", type=Path)
p.add_argument("--report", type=Path, required=True)
p.add_argument("--require-matched", action="store_true")
p.add_argument("--max-rate", type=float)
p.add_argument("--approval-manifest", type=Path)
a = p.parse_args()
d = json.loads(a.thresholds.read_text())
cells = []
for c in d["cells"]:
    q = c.get("diagnostics", {})
    cells.append({k: c[k] for k in ("arm", "topology", "agents")} | q)
flagged = [c for c in cells if not c.get("within_tolerance", c.get("converged", False))]
out = {"alpha_requested": d["alpha"], "cells": cells, "n_cells": len(cells),
       "n_matched": len(cells)-len(flagged), "flagged": flagged,
       "all_matched": bool(cells) and not flagged,
       "interpretation": "Equal achieved FPR may be claimed only for matched cells; retain flagged cells as explicit unmatched simulations."}
a.report.parent.mkdir(parents=True, exist_ok=True)
a.report.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
if a.require_matched and not out["all_matched"]:
    raise SystemExit("Calibration did not achieve the requested alpha in every cell; see the report.")
if a.max_rate is not None and any(c.get("rate", float("inf")) > a.max_rate for c in cells):
    raise SystemExit("A calibration cell exceeds the explicitly allowed empirical rate cap.")
if a.approval_manifest is not None:
    approval = json.loads(a.approval_manifest.read_text())
    allowed = {tuple(c) for c in approval["allowed_flagged_cells"]}
    unexpected = [c for c in flagged if (c["arm"], c["topology"], c["agents"]) not in allowed]
    if unexpected:
        raise SystemExit("Unmatched cells were not explicitly approved: " + json.dumps(unexpected))
