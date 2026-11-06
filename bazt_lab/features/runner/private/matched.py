"""D116: the matched-timing controls act only at the rounds where BAZT intervened in the same unit scenario.

A block that runs `*_matched` arms names its timing source (`Block.timing_source`: mock_legacy reads mock_frontier). For
each matched unit the source unit is the same unit with arm BAZT and the source block's name (every other field equal:
cell, topology, family, agents, seed, sink and any variant field). Its committed row (`core.committed_row_files` of the
source block) holds `intervened_by_round`, T booleans with entry r = round r+1, which the unit is handed unchanged as
`timing_schedule`; an attacked unit also gets the record of the attack-free source unit as `twin_timing_schedule` (the
mock simulates its attack-free twin with it, D109). Both are annotations: they take no part in the unit id.

Everything is read before the launch writes anything: a scenario the source block never runs BAZT at, or one whose
BAZT row is not committed yet, is refused naming the block and an example, and so are BAZT rows produced under other
settings than the launch's (the source block's settings.json against its view of them). The source is pinned in the block directory
(`timing_schedules.json`: the source block, the settings its BAZT rows were produced under and a digest of the rounds
read), so a launch after BAZT was run again with other settings or other rounds is refused instead of mixing rows.

Reading: the rows files of one Platform.run_batch hold one group (one arm), so a file whose first row is not BAZT, or
not of a group the block needs, is skipped unread. A row that sat in a skipped file is reported missing, never
mistaken: the refusal is the failure mode.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping

from bazt_lab.core import public as core
from bazt_lab.features.runner.private import identity, needs
from bazt_lab.features.runner.private import units as u

SOURCE_ARM = "BAZT"
COLUMN = "intervened_by_round"
FIELD, TWIN_FIELD = "timing_schedule", "twin_timing_schedule"
PIN = "timing_schedules.json"
THRESHOLDS_KEY = "mock.legacy.matched_thresholds"
THRESHOLD_CHOICES = ("base", "own")


def is_matched(arm: str) -> bool:
    return arm.endswith(needs.MATCHED_SUFFIX)


def base_arm(arm: str) -> str:
    return arm[:-len(needs.MATCHED_SUFFIX)]


def source_unit(unit: Mapping, source: str, attack: str | None = None) -> dict:
    """The BAZT unit of the same scenario in the source block (annotations dropped)."""
    plain = {k: v for k, v in unit.items() if k not in u.ANNOTATIONS}
    plain.update(arm=SOURCE_ARM, block=source)
    if attack is not None:
        plain["attack"] = attack
    return plain


def _holds(grid: u.MockGrid, unit: Mapping) -> bool:
    """Whether the source grid has this BAZT unit (closed form, the grid is never expanded)."""
    if (SOURCE_ARM, (unit["audit"], unit["intervention"])) not in [tuple(x) for x in grid.arms_cells]:
        return False
    if (unit["topology"] not in grid.topologies or unit["attack"] not in grid.families
            or unit["agents"] not in grid.agent_counts or unit["seed"] not in grid.seeds):
        return False
    if unit["sink"] not in (range(unit["agents"]) if grid.every_sink else (None,)):
        return False
    return any(grid._unit(SOURCE_ARM, unit["audit"], unit["intervention"], unit["topology"], unit["attack"],
                          unit["agents"], unit["seed"], unit["sink"], grid._extra(v)) == dict(unit)
               for v in grid.variants)


def _describe(unit: Mapping) -> str:
    keys = ("audit", "intervention", "topology", "attack", "agents", "seed", "sink", "preset", "condition",
            "cost_condition")
    return ", ".join(f"{k}={unit[k]}" for k in keys if k in unit)


def _schedule(row: Mapping, source: str) -> tuple:
    value = row.get(COLUMN)
    if not isinstance(value, list) or not value or not all(type(x) is bool for x in value):
        raise ValueError(f"the BAZT row of unit {row.get('id')!r} in {source} has no {COLUMN} (T booleans) but "
                         f"{value!r}: it was produced before D116 recorded the intervention rounds; run --block "
                         f"{source} again in a fresh results root")
    return tuple(value)


class Schedules:
    """The BAZT records the block's matched units replay: `annotate(unit)` adds them; `pin()` is what the block pins."""

    def __init__(self, source: str, by_id: Mapping[str, tuple], settings: Mapping) -> None:
        self.source, self.by_id, self.settings = source, dict(by_id), dict(settings)

    def annotate(self, unit: Mapping) -> dict:
        if not is_matched(unit["arm"]):
            return dict(unit)
        out = {**unit, FIELD: list(self.by_id[u.unit_id(source_unit(unit, self.source))])}
        if unit["attack"] != "none":
            out[TWIN_FIELD] = list(self.by_id[u.unit_id(source_unit(unit, self.source, attack="none"))])
        return out

    def pin(self) -> dict:
        text = json.dumps(sorted(self.by_id.items()), separators=(",", ":"))
        return {"source": self.source, "arm": SOURCE_ARM, "units": len(self.by_id), "settings": self.settings,
                "rounds_sha256": hashlib.sha256(text.encode()).hexdigest()}


def load(s: Mapping, name: str, grid, source: str, source_grid: u.MockGrid, root: Path) -> Schedules:
    """The BAZT records of every matched unit of the block, refused (ValueError) before anything is written."""
    needed: Dict[str, dict] = {}
    for unit in grid:
        if is_matched(unit["arm"]):
            for attack in {None, "none"} if unit["attack"] != "none" else {None}:
                src = source_unit(unit, source, attack)
                needed.setdefault(u.unit_id(src), src)
    never = [v for v in needed.values() if not _holds(source_grid, v)]
    if never:
        raise ValueError(f"its matched-timing controls replay BAZT's intervention rounds of the same unit (D116), and "
                         f"{source} never runs BAZT at {len(never)} of the {len(needed)} unit(s) they need, e.g. "
                         f"{_describe(never[0])}: give {name} only cells, topologies, families, agents and seeds at "
                         f"which {source} runs BAZT")
    directory = Path(root) / s["profile"] / source
    produced = identity.read(directory / "settings.json")
    if produced is None or not (directory / "units").is_dir():
        raise ValueError(f"its matched-timing controls replay BAZT's intervention rounds of the same unit (D116), and "
                         f"{directory} holds no BAZT rows: run --block {source} first")
    current = identity.block_view(s, source, "mock")          # a timing source is a mock block
    if produced != current:
        raise ValueError(f"its matched-timing controls replay BAZT's intervention rounds (D116), and the BAZT rows of "
                         f"{source} were produced under other settings than this launch: "
                         f"{identity.differences(produced, current)}; run both blocks under one setting")
    groups = {u.group_key(v) for v in needed.values()}
    found: Dict[str, tuple] = {}
    files, _ = core.committed_row_files(directory)
    for path in files:
        with path.open(encoding="utf-8") as f:
            first = f.readline()
        if not first.strip():
            continue
        head = json.loads(first)
        if head.get("arm") != SOURCE_ARM or u.group_key(head) not in groups:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("id")
            if uid not in needed:
                continue
            rounds = _schedule(row, source)
            if uid in found and found[uid] != rounds:
                raise ValueError(f"{directory} holds two committed BAZT rows of unit {uid} with different "
                                 f"{COLUMN}: rerun {source} in a fresh results root")
            found[uid] = rounds
    missing = [v for k, v in needed.items() if k not in found]
    if missing:
        raise ValueError(f"BAZT has no committed row for {len(missing)} of the {len(needed)} unit(s) whose intervention "
                         f"rounds the matched-timing controls replay (D116), e.g. {_describe(missing[0])}: run --block "
                         f"{source} first (BAZT runs before {name}; a crashed launch resumes)")
    return Schedules(source, found, produced)


def check_pin(path: Path, schedules: Schedules) -> None:
    """A block that pinned another BAZT source than the one read now is refused (ValueError)."""
    if not path.exists():
        return
    saved, now = json.loads(path.read_text(encoding="utf-8")), schedules.pin()
    if saved == now:
        return
    if saved.get("settings") != now["settings"]:
        what = f"BAZT rows produced under other settings: {identity.differences(saved.get('settings', {}), now['settings'])}"
    elif saved.get("source") != now["source"]:
        what = f"another source block ({saved.get('source')!r} -> {now['source']!r})"
    else:
        what = (f"other BAZT rounds ({saved.get('units')} -> {now['units']} units, rounds_sha256 "
                f"{str(saved.get('rounds_sha256'))[:12]} -> {now['rounds_sha256'][:12]})")
    raise ValueError(f"{path} pins the BAZT intervention rounds its rows replay, and {schedules.source} now holds "
                     f"{what}: rows of two schedules would mix; use another --results-root or --profile")


def threshold_choice(s: Mapping) -> str:
    """The owner's choice for a calibrated matched arm (core.Unset while unset): 'base', the thresholds calibrated for the
    legacy arm it wraps in the same cell; 'own', its own calibration (open loop only: on the attack-free tune split
    there is no BAZT schedule, so the closed loop cannot run it)."""
    value = core.get(s, THRESHOLDS_KEY)
    if value not in THRESHOLD_CHOICES:
        raise ValueError(f"{THRESHOLDS_KEY} must be one of {list(THRESHOLD_CHOICES)}, got {value!r}")
    return value
