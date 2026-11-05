"""`status`: one screen with the progress of every block, the owner decisions still unset (with their options, taken
from the comments of profiles.toml) and whether the secrets are present."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping

from bazt_lab.core import public as core
from bazt_lab.features.runner.adapters import env
from bazt_lab.features.runner.private import ledger, needs, thresholds
from bazt_lab.features.runner.private.blocks import BLOCKS, CALIBRATION_BLOCK, Block
from bazt_lab.features.runner.private.runtime import PRE_REGISTERED, RUNTIME_BY_PLATFORM

ADAPTIVE_KEYS = ("search_seeds", "insider_sets", "start_fractions", "rates", "detectabilities", "evasion_cost",
                 "max_candidates")
PLACEHOLDER = re.compile(r"^#\s*([A-Za-z0-9_.\"]+)\s*=\s*(.*)$")
HEADER = re.compile(r"^#?\s*\[([A-Za-z0-9_.]+)\]")


def options(path: Path = core.SETTINGS_PATH) -> Dict[str, str]:
    """{dotted key: comment} for every key that profiles.toml lists commented out (unset), with the comment lines
    that continue under it."""
    found: Dict[str, str] = {}
    table, last = "", None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        header = HEADER.match(line)
        if header and "=" not in line:
            table, last = header.group(1), None
            continue
        match = PLACEHOLDER.match(line)
        if match:
            key = (f"{table}." if table else "") + match.group(1).strip('"')
            _, _, comment = match.group(2).partition("#")
            found[key], last = comment.strip(), key
        elif last and re.match(r"^\s{20,}#", line):
            found[last] = (found[last] + " " + line.strip().lstrip("# ")).strip()
        else:
            last = None
    return found


def needed(s: Mapping, block: Block) -> List[str]:
    """Settings a launch of the block would need, in the order it would meet them."""
    keys = [f"runtime.{k}" for k in (*PRE_REGISTERED, *RUNTIME_BY_PLATFORM[block.platform])]
    if block.platform == "llm":
        keys += [f"runtime.llm_sink_positions.{t}" for t in core.get(s, "llm.topologies")]
    try:
        grid = block.units(s, False)
    except core.Unset as e:
        return keys + [e.args[0]]
    except SystemExit:
        return keys
    try:
        keys += needs.required_keys(s, block.platform, grid.arms(), block.name)
    except core.Unset as e:
        keys.append(e.args[0])
    if block.kind == "calibrate":
        keys.append("calibration.mode")
    if block.adaptive:
        keys += [f"mock.adaptive.{k}" for k in ADAPTIVE_KEYS]
        keys += [f"mock.adaptive.knowledge.{c}" for c in core.get(s, "mock.adaptive.conditions")]
    return keys


def done_count(directory: Path) -> int:
    units = directory / "units"
    return sum(1 for e in os.scandir(units) if e.name.endswith(".done")) if units.is_dir() else 0


def calibration_warnings(results: Path) -> List[str]:
    """Cells whose calibration did not converge, or whose worst episode rate is above 5 x alpha."""
    out = []
    for block in sorted(set(CALIBRATION_BLOCK.values())):
        path = results / block / thresholds.FILE
        if not path.exists():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        for cell in saved["cells"]:
            diag = cell.get("diagnostics") or {}
            label = thresholds.describe((cell["arm"], cell["topology"], cell["agents"], cell["backbone"], cell["task"]))
            if diag.get("converged") is False or diag.get("within_tolerance") is False:
                out.append(f"{block}: {label}: did not converge (passes {diag.get('passes')}, rate {diag.get('rate')})")
            worst = diag.get("worst_episode_rate")
            if worst is not None and worst > 5 * saved["alpha"]:
                out.append(f"{block}: {label}: worst_episode_rate {worst} is above 5 x alpha ({5 * saved['alpha']:g})")
    return out


def render(s: Mapping, root: Path, settings_path: Path = core.SETTINGS_PATH) -> str:
    root, profile = Path(root), s["profile"]
    errors, comments = ledger.read(root, profile), options(settings_path)
    lines = [f"profile {profile}   results {root / profile}", "",
             f"{'block':<26}{'platform':<9}{'done/total':>16}  state"]
    unset: Dict[str, List[str]] = {}
    for name, block in BLOCKS.items():
        try:
            total = f"{len(block.units(s, False)):,}"
        except core.Unset:
            total = "-"
        except SystemExit:
            total = "?"
        done = done_count(root / profile / name)
        state = ("scope not set" if total == "-" else "done" if total != "?" and f"{done:,}" == total else
                 "running or stopped" if done else "not started")
        if name in errors:
            state += f"   last error: {errors[name]}"
        lines.append(f"{name:<26}{block.platform:<9}{f'{done:,}/{total}':>16}  {state}")
        unset[name] = [k for k in dict.fromkeys(needed(s, block)) if not core.is_set(s, k)]
    lines += ["", "owner decisions still unset"]
    shown = set()
    for name, keys in unset.items():
        if not keys:
            continue
        lines.append(f"  {name}")
        for key in keys:
            first = key not in shown
            shown.add(key)
            note = comments.get(key) if first else "(listed above)"
            lines.append(f"    {key}" + (f"   {note}" if note else ""))
    if not shown:
        lines.append("  none")
    warnings = calibration_warnings(root / profile)
    if warnings:
        lines += ["", "calibration warnings"] + [f"  {w}" for w in warnings]
    lines += ["", "secrets (presence only)"]
    lines += [f"  {name:<20}{'present' if ok else 'missing'}" for name, ok in env.present().items()]
    return "\n".join(lines)
