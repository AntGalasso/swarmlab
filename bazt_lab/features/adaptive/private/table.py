"""The best-response table: the attack_params the adaptive attacker plays, per cell of the mock_adaptive block.

A cell is every field of an evaluation unit except the episode (seed, sink) and the runner's annotations, so the same
cell identifies the search units and the evaluation units: arm, budget cell, topology, family, N, condition, ...
An entry covers a whole cell, one seed of it, or one (seed, sink) episode (the knowledge level of the condition).

On disk: `meta.json` (settings fingerprint) and one small file per finished cell, written atomically, so a crashed
search keeps every finished cell and a rerun with other settings is refused instead of mixing results.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

META = "meta.json"
VERSION = 1
NOT_CELL = frozenset({"block", "seed", "sink", "id", "thresholds", "attack_params"})
ATTACK_FREE = "none"


def cell_of(unit: Mapping) -> dict:
    return {k: v for k, v in unit.items() if k not in NOT_CELL}


def cell_id(cell: Mapping) -> str:
    """Canonical text of a cell. Floats round-trip exactly through json, so 0.1 + 0.2 stays 0.30000000000000004."""
    return json.dumps(cell, sort_keys=True)


@dataclass(frozen=True)
class Choice:
    attack_params: dict
    asr: float              # mean final-round ASR of the chosen strategy over the search episodes of its key
    baseline_asr: float     # the same for the family's default attack ({})
    candidates: int         # size of the strategy space searched
    episodes: int           # search episodes behind each of those means


class BestResponses:
    def __init__(self) -> None:
        self._by_key: Dict[tuple, Choice] = {}

    def __len__(self) -> int:
        return len(self._by_key)

    def add(self, cell: Mapping, seed: Optional[int], sink: Optional[int], choice: Choice) -> None:
        self._by_key[(cell_id(cell), seed, sink)] = Choice(copy.deepcopy(choice.attack_params), choice.asr,
                                                           choice.baseline_asr, choice.candidates, choice.episodes)

    def choice(self, unit: Mapping) -> Choice:
        """The entry of the unit's cell at the finest granularity stored: (seed, sink), then seed, then the cell."""
        cid = cell_id(cell_of(unit))
        seed, sink = unit.get("seed"), unit.get("sink")
        for key in ((cid, seed, sink), (cid, seed, None), (cid, None, None)):
            if key in self._by_key:
                return self._by_key[key]
        raise LookupError(f"no best response for cell {cid} (seed={seed}, sink={sink}); run the adaptive search first")

    def params_for(self, unit: Mapping) -> dict:
        if unit.get("attack") == ATTACK_FREE:
            return {}
        return copy.deepcopy(self.choice(unit).attack_params)

    def annotate(self, unit: Mapping) -> dict:
        return {**unit, "attack_params": self.params_for(unit)}

    def missing(self, units: Iterable[Mapping]) -> List[dict]:
        """Each cell with at least one unit that has no entry, once, in first-seen order."""
        seen: Dict[str, dict] = {}
        for u in units:
            try:
                self.params_for(u)
            except LookupError:
                cell = cell_of(u)
                seen.setdefault(cell_id(cell), cell)
        return list(seen.values())

    @classmethod
    def load(cls, directory, fingerprint: str) -> BestResponses:
        """The table of `directory`, refused if it was produced with other settings than `fingerprint`."""
        directory = Path(directory)
        if not (directory / META).exists():
            raise FileNotFoundError(f"{directory / META} does not exist: run the adaptive search first")
        check_fingerprint(directory, fingerprint)
        table = cls()
        for path in sorted(directory.glob("*.json")):
            if path.name == META:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for r in data["records"]:
                table.add(data["cell"], r["seed"], r["sink"],
                          Choice(r["attack_params"], r["asr"], r["baseline_asr"], r["candidates"], r["episodes"]))
        return table


def _write_json(path: Path, obj) -> None:
    """Atomic: a reader sees the old file or the whole new one."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _cell_path(directory, cell: Mapping) -> Path:
    return Path(directory) / f"{hashlib.sha256(cell_id(cell).encode()).hexdigest()[:16]}.json"


def check_fingerprint(directory, fingerprint: str) -> None:
    found = json.loads((Path(directory) / META).read_text(encoding="utf-8"))["fingerprint"]
    if found != fingerprint:
        raise ValueError(f"{directory} holds best responses produced with other settings (fingerprint {found}, "
                         f"now {fingerprint}); delete it or use another output directory")


def ensure_meta(directory, fingerprint: str, space: Mapping) -> None:
    """Create the table directory with its fingerprint, or check that an existing one has the same."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / META).exists():
        check_fingerprint(directory, fingerprint)
        return
    _write_json(directory / META, {"version": VERSION, "fingerprint": fingerprint, "space": space})


def has_cell(directory, cell: Mapping, inputs: str) -> bool:
    """Whether the cell is finished. `inputs` identifies what the search read outside the settings (the cell's
    calibrated thresholds); a cell finished against other inputs is refused, not silently reused."""
    path = _cell_path(directory, cell)
    if not path.exists():
        return False
    found = json.loads(path.read_text(encoding="utf-8"))["inputs"]
    if found != inputs:
        raise ValueError(f"{path} was searched against other thresholds ({found}, now {inputs}); delete it to "
                         f"search the cell again")
    return True


def write_cell(directory, cell: Mapping, records: List[dict], inputs: str) -> None:
    """records: dicts with seed, sink, attack_params, asr, baseline_asr, candidates, episodes."""
    _write_json(_cell_path(directory, cell), {"cell": dict(cell), "inputs": inputs, "records": records})


def read_records(directory, cell: Mapping) -> List[dict]:
    """The records of a finished cell, [] if it has none."""
    path = _cell_path(directory, cell)
    return json.loads(path.read_text(encoding="utf-8"))["records"] if path.exists() else []


def covers(directory, cell: Mapping, keys: Iterable[tuple]) -> bool:
    """Whether the finished cell has an entry for every (seed, sink) key (a cell-level key is (None, None))."""
    have = {(r["seed"], r["sink"]) for r in read_records(directory, cell)}
    return all(k in have for k in keys)
