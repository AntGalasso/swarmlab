"""`status`: one screen with the progress of every block, the owner decisions still unset (with their options, taken
from the comments of profiles.toml) and whether the secrets are present."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Dict, List, Mapping

from bazt_lab.core import public as core
from bazt_lab.features.runner.adapters import env
from bazt_lab.features.runner.private import ledger, needs, thresholds
from bazt_lab.features.runner.private.blocks import BLOCKS, CALIBRATION_BLOCK, Block, unrunnable
from bazt_lab.features.runner.private.runtime import PRE_REGISTERED, RUNTIME_BY_PLATFORM
from bazt_lab.features.runner.private.wiring import Wiring

ADAPTIVE_KEYS = ("search_seeds", "insider_sets", "start_fractions", "rates", "detectabilities", "evasion_cost",
                 "max_candidates")
PLACEHOLDER = re.compile(r"^#\s*([A-Za-z0-9_.\"]+)\s*=\s*(.*)$")
# A table header, active or commented out: `[key]` alone on its line, then at most a comment (which may hold anything).
HEADER = re.compile(r"^\s*(#\s*)?\[(?!\[)([^\[\]]+)\]\s*(?:#.*)?$")


def _table_path(key: str) -> List[str] | None:
    """The parts of a table header's key as TOML reads it (quotes removed, spaces around dots ignored); None when the
    text is no TOML key."""
    try:
        node = tomllib.loads(f"[{key}]")
    except tomllib.TOMLDecodeError:
        return None
    parts = []
    while node:
        ((name, node),) = node.items()
        parts.append(name)
    return parts


def _is_table(tree: Mapping, parts: List[str]) -> bool:
    for part in parts:
        if not isinstance(tree, Mapping) or part not in tree:
            return False
        tree = tree[part]
    return isinstance(tree, Mapping)


def options(path: Path = core.SETTINGS_PATH) -> Dict[str, str]:
    """{dotted key: comment} for every key that profiles.toml lists commented out (unset), with the comment lines
    that continue under it. An active header must be a table of the parsed file, so a `[1]` line inside a multi-line
    array is not taken for one."""
    found: Dict[str, str] = {}
    table, last = "", None
    text = Path(path).read_text(encoding="utf-8")
    tree = tomllib.loads(text)
    for line in text.splitlines():
        header = HEADER.match(line)
        parts = _table_path(header.group(2)) if header else None
        if parts and (header.group(1) or _is_table(tree, parts)):
            table, last = ".".join(parts), None
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


SCOPED = ("epidemiology", "ablations", "adaptive", "presets_robustness", "costs_robustness", "legacy")


def relevant(key: str, block: Block, arms: set) -> bool:
    """Whether a block will demand a setting that profiles.toml lists as unset, judged from the block, its platform,
    its scope section and its arms."""
    head, _, rest = key.partition(".")
    families = {needs.family(a) for a in arms}
    if head in ("runtime", "tracking", "sync", "analysis", "conventional") or key.startswith("llm.judge."):
        return False                      # run-time variables and the judge/trust keys come from needed() and needs.py
    if head == "bazt":
        return bool(families & {"BAZT", needs.UPSTREAM})
    if head == "calibration":
        return block.kind == "calibrate"
    if key.startswith("costs.robustness."):
        return block.name == "mock_costs_robustness"
    if head == "auditor_presets":
        return block.name == "mock_presets_robustness"
    if head == "gnn_train":
        return block.name == "gnn_train"
    if head == "gnn":
        return bool(families & set(needs.GNN_ARMS)) and block.name != "gnn_train"
    if key.startswith("llm.adaptive_check."):
        return block.name == "llm_adaptive_check"
    if key == "llm.calibration_conversations_per_cell":
        return block.name == "llm_calibration"
    if head == "llm":
        return block.platform == "llm"
    if head == "mock":
        section = rest.split(".")[0]
        return block.name == f"mock_{section}" if section in SCOPED else block.platform == "mock"
    return False


def needed(s: Mapping, block: Block, comments: Mapping = ()) -> List[str]:
    """Settings a launch of the block would need, in the order it would meet them."""
    keys = [f"runtime.{k}" for k in (*PRE_REGISTERED, *RUNTIME_BY_PLATFORM[block.platform])]
    if block.platform == "llm":
        keys += [f"runtime.llm_sink_positions.{t}" for t in core.get(s, "llm.topologies")]
    arms: set = set()
    try:
        grid = block.units(s, False)
        arms = grid.arms()
    except core.Unset as e:
        grid = None
        keys.append(e.args[0])
    except SystemExit:
        grid = None
    if grid is not None:
        try:
            keys += needs.required_keys(s, block.platform, arms, block.name)
        except core.Unset as e:
            keys.append(e.args[0])
        if block.kind == "calibrate":
            keys.append("calibration.mode")
        if block.adaptive:
            keys += [f"mock.adaptive.{k}" for k in ADAPTIVE_KEYS]
            keys += [f"mock.adaptive.knowledge.{c}" for c in core.get(s, "mock.adaptive.conditions")]
    return keys + [k for k in comments if relevant(k, block, arms)]


def done_count(directory: Path) -> int:
    """Finished units, from the store's own markers (per unit, or committed in batches). Nothing is created."""
    return core.CheckpointStore(directory).done_count() if (directory / "units").is_dir() else 0


def calibrated_cells(directory: Path) -> int:
    path = directory / thresholds.FILE
    return len(json.loads(path.read_text(encoding="utf-8"))["cells"]) if path.exists() else 0


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


def render(s: Mapping, root: Path, settings_path: Path = core.SETTINGS_PATH, *, wiring: Wiring) -> str:
    """The status table. `wiring` answers what a platform runs, for a block whose units carry an adaptive condition."""
    root, profile = Path(root), s["profile"]
    errors, comments = ledger.read(root, profile), options(settings_path)
    left_out = ledger.read_skipped(root, profile)
    compute = (core.get(s, "compute.device") if core.is_set(s, "compute.device") else "cpu",
               core.is_set(s, "compute.compile") and core.get(s, "compute.compile"))
    lines = [f"profile {profile}   results {root / profile}   device {compute[0]}" + (" (compiled)" if compute[1] else ""),
             "",
             f"{'block':<26}{'platform':<9}{'done/total':>16}  state"]
    unset: Dict[str, List[str]] = {}
    for name, block in BLOCKS.items():
        try:
            grid = block.units(s, False)
            total = f"{len(grid.cells()) if block.kind == 'calibrate' else len(grid):,}"       # cells for a calibration
        except core.Unset:
            total = "-"
        except SystemExit:
            total = "?"
        done = (calibrated_cells(root / profile / name) if block.kind == "calibrate"
                else done_count(root / profile / name))
        if name in left_out and total not in ("-", "?"):
            total = f"{int(total.replace(',', '')) - left_out[name]['units']:,}"
        state = ("scope not set" if total == "-" else "done" if total != "?" and f"{done:,}" == total else
                 "running or stopped" if done else "not started")
        reason = unrunnable(block, s, wiring) if total != "-" else None
        if reason:
            state = f"not runnable: {reason}"
        if name in left_out:
            state += f"   ({left_out[name]['units']} units skipped: no GNN for " + "; ".join(
                c.split(": ", 1)[1] for c in left_out[name]["cells"]) + ")"
        if name in errors:
            state += f"   last error: {errors[name]}"
        lines.append(f"{name:<26}{block.platform:<9}{f'{done:,}/{total}':>16}  {state}")
        unset[name] = [k for k in dict.fromkeys(needed(s, block, comments)) if not core.is_set(s, k)]
    unset["analysis"] = [k for k in comments if k.startswith("analysis.") and not core.is_set(s, k)]
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
