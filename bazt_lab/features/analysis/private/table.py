"""Result rows as compact columns: int-coded unit fields, float64 metrics, rows grouped by cell, paired by episode.

Rows come from the committed rows files of <results_dir>/<block>/units/ and of its worker shards units/<shard>/
(`core.committed_row_files`, the store's done rule), one `{"unit": {...}, **metrics}` per line (unit fields may also be
repeated flat in the row; the row wins a collision). A full mock frontier has millions of rows, so a row is never kept as a dict: every unit
field is a dictionary code, every metric an array('d').

Vocabularies are sorted after loading, so codes, groups and pair order do not depend on file or row order.
A *group* is the set of rows of one (block, arm, audit, intervention, topology, agents, condition, preset,
cost_condition, sanitize_mode, ma_check, attack_free); the block is the folder the rows were read from. Inside a
group a row is identified by its *episode*: (attack, seed, sink, conversation, task, backbone). Two groups are paired
by an inner join on the episode id; a repeated episode inside a group is an error, never a silent pick.
"""
from __future__ import annotations

import json
import math
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence

import numpy as np

from bazt_lab.core import public as core
from bazt_lab.features.metrics.public import recovery_time_accuracy

BUDGET_FIELDS = ("audit", "intervention")
UNIT_FIELDS = ("arm", *BUDGET_FIELDS, "topology", "agents", "condition", "preset", "cost_condition", "sanitize_mode",
               "ma_check")
EPISODE_FIELDS = ("attack", "seed", "sink", "conversation", "task", "backbone")
RAW_FIELDS = UNIT_FIELDS + EPISODE_FIELDS
UNCAPPED = "uncapped"


class Key(NamedTuple):
    """A group. Compares equal to the plain tuple of its fields."""
    block: str
    arm: str | None
    audit: float | str | None
    intervention: float | str | None
    topology: str | None
    agents: int | None
    condition: str | None
    preset: str | None
    cost_condition: str | None
    sanitize_mode: str | None
    ma_check: str | None
    attack_free: bool | None

    @property
    def ctx(self) -> Key:
        """The same cell without the arm and the attack flag: what two arms are compared in."""
        return self._replace(arm=None, attack_free=None)

    def with_arm(self, arm: str, attack_free: bool | None = None) -> Key:
        return self._replace(arm=arm, attack_free=self.attack_free if attack_free is None else attack_free)


class DuplicateRows(ValueError):
    """The same episode appears twice in one group (a unit written twice, or two blocks overlapping)."""


def budget_order(level) -> float:
    """Sort key of a budget level: a fraction of deliveries, "uncapped" last."""
    return math.inf if level == UNCAPPED else float(level)


def _normalise(name: str, value):
    if name in BUDGET_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def _scalar_key(value):
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, (bool, int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))


def sort_key(values: Sequence) -> tuple:
    """A total order over tuples of None, numbers and strings (a budget level is a number or "uncapped")."""
    return tuple(_scalar_key(v) for v in values)


@dataclass
class Table:
    n: int
    vocab: Dict[str, list]
    codes: Dict[str, np.ndarray]
    metrics: Dict[str, np.ndarray]
    episode_id: np.ndarray
    index: Dict[Key, np.ndarray]             # group key -> row numbers, ascending episode id
    missing_blocks: List[str] = field(default_factory=list)
    torn_lines: int = 0
    empty_metrics: List[str] = field(default_factory=list)
    uncommitted_files: int = 0               # rows files without their done marker, not read (core.committed_row_files)
    _modeless: Dict | None = field(default=None, repr=False)

    def groups(self) -> Dict[Key, np.ndarray]:
        return self.index

    def resolve(self, key: Key) -> Key:
        """`key`, or the one group that differs from it only in `sanitize_mode`. An arm that never sanitizes (A, B, C)
        runs once, in the first mode, and serves every sanitize mode of an arm that does."""
        if key in self.index:
            return key
        if self._modeless is None:
            self._modeless = {}
            for k in self.index:
                self._modeless.setdefault(k._replace(sanitize_mode=None), []).append(k)
        alternatives = self._modeless.get(key._replace(sanitize_mode=None), [])
        return alternatives[0] if len(alternatives) == 1 else key

    def has(self, key: Key) -> bool:
        return self.resolve(key) in self.index

    def aligned(self, keys: Sequence[tuple]):
        """Row numbers of the episodes present in every group, one array per key and in the same episode order;
        None if one of the groups is absent. A group missing only in its sanitize mode is served by `resolve`."""
        keys = [self.resolve(k) if isinstance(k, Key) else k for k in keys]
        if any(k not in self.index for k in keys):
            return None
        rows = [self.index[k] for k in keys]
        ids = [self.episode_id[r] for r in rows]
        common = ids[0]
        for other in ids[1:]:
            common = np.intersect1d(common, other, assume_unique=True)
        return [r[np.searchsorted(i, common)] for r, i in zip(rows, ids, strict=True)]


CLEAN_COLUMNS = ("accuracy_clean_reference", "nrp", "recovery_time_accuracy")


class _Builder:
    def __init__(self, metric_columns: Sequence[str], pair_clean: bool = False) -> None:
        extra = ("asr_final", "accuracy_clean", *CLEAN_COLUMNS) if pair_clean else ()
        self.metric_columns = tuple(dict.fromkeys((*metric_columns, *extra)))
        self.by_round: List = [] if pair_clean else None
        self.maps: Dict[str, dict] = {f: {} for f in ("block", *RAW_FIELDS)}
        self.codes = {f: array("i") for f in self.maps}
        self.attack_free = array("b")
        self.metrics = {m: array("d") for m in self.metric_columns}

    def add(self, raw: dict, block: str) -> None:
        unit = raw.get("unit")
        row = {**unit, **{k: v for k, v in raw.items() if k != "unit"}} if isinstance(unit, dict) else raw
        for f, m in self.maps.items():
            value = block if f == "block" else _normalise(f, row.get(f))
            self.codes[f].append(m.setdefault(value, len(m)))
        free = row.get("attack") == "none"
        if "attack_free" in row and bool(row["attack_free"]) != free:
            raise ValueError(f"attack_free={row['attack_free']!r} contradicts attack={row.get('attack')!r} "
                             f"in a row of arm {row.get('arm')!r}")
        self.attack_free.append(int(free))
        if self.by_round is not None:
            self.by_round.append(row.get("accuracy_by_round"))
        for m in self.metric_columns:
            v = row.get(m)
            try:
                self.metrics[m].append(math.nan if v is None else float(v))
            except (TypeError, ValueError):         # a list (a per-round column) or a text: never a metric value
                raise ValueError(f"metric {m!r} of a row of arm {row.get('arm')!r} in block {block!r} is not a number: "
                                 f"{v!r}") from None

    def finish(self, missing_blocks: List[str], torn: int, uncommitted: int) -> Table:
        n = len(self.attack_free)
        vocab: Dict[str, list] = {}
        codes: Dict[str, np.ndarray] = {}
        for f, m in self.maps.items():
            values = sorted(m, key=_scalar_key)
            rank = np.empty(len(values), dtype=np.int64)
            for new, v in enumerate(values):
                rank[m[v]] = new
            vocab[f] = values
            codes[f] = rank[np.frombuffer(self.codes[f], dtype=np.int32)] if n else np.empty(0, dtype=np.int64)
        vocab["attack_free"] = [False, True]
        codes["attack_free"] = np.frombuffer(self.attack_free, dtype=np.int8).astype(np.int64) if n \
            else np.empty(0, dtype=np.int64)
        metrics = {m: np.frombuffer(a, dtype=np.float64) if n else np.empty(0) for m, a in self.metrics.items()}
        if math.prod(max(len(vocab[f]), 1) for f in EPISODE_FIELDS) >= 2 ** 62:
            raise ValueError("too many distinct episode field values to pack into one int64 id")
        episode_id = np.zeros(n, dtype=np.int64)
        for f in EPISODE_FIELDS:
            episode_id = episode_id * max(len(vocab[f]), 1) + codes[f]
        table = Table(n, vocab, codes, metrics, episode_id, {}, missing_blocks, torn,
                      [m for m, v in metrics.items() if n == 0 or np.isnan(v).all()], uncommitted)
        table.index = _group(table)
        return table

    def pair_clean(self, t: Table, margin: float) -> None:
        """Fill the clean reference, NRP and recovery time of every attack row the platform could not pair (NaN
        reference) from the attack-free row of the same unit and episode. A missing twin is an error."""
        if self.by_round is None or not t.n:
            return
        fields = [f for f in ("block", *RAW_FIELDS) if f != "attack"]       # the twin has the same ma_check
        ident = np.stack([t.codes[f] for f in fields], axis=1)
        free = t.codes["attack_free"] == 1
        twin = {tuple(ident[i]): t.metrics["accuracy_clean"][i] for i in np.flatnonzero(free)}
        # An adaptive block runs no attack-free units: its twin is the plain run (no condition) of any loaded block.
        at_block, at_cond = fields.index("block"), fields.index("condition")
        none_code = t.vocab["condition"].index(None) if None in t.vocab["condition"] else -1

        def plain(k: tuple) -> tuple:
            return (*k[:at_block], *k[at_block + 1:at_cond], none_code, *k[at_cond + 1:])

        by_plain = {plain(k): v for k, v in twin.items() if k[at_cond] == none_code}
        ref, nrp, rec = (t.metrics[c] for c in CLEAN_COLUMNS)
        for i in np.flatnonzero(~free & np.isnan(ref)):
            k = tuple(ident[i])
            clean = twin.get(k, by_plain.get(plain(k)))
            if clean is None:
                unit = {f: t.vocab[f][t.codes[f][i]] for f in ("block", *UNIT_FIELDS, *EPISODE_FIELDS) if f != "attack"}
                raise ValueError(f"no attack-free twin for the attack row {unit}; the clean reference, NRP and "
                                 "recovery time need the same unit without the attack")
            ref[i] = clean
            nrp[i] = clean * (1.0 - t.metrics["asr_final"][i])
            acc = self.by_round[i]
            rec[i] = math.nan if not acc else recovery_time_accuracy(acc, clean, margin)


GROUP_FIELDS = Key._fields


def _group(t: Table) -> Dict[Key, np.ndarray]:
    if t.n == 0:
        return {}
    cols = [t.codes[f] for f in GROUP_FIELDS]
    order = np.lexsort([t.episode_id] + cols[::-1])
    sorted_cols = np.stack([c[order] for c in cols])
    ids = t.episode_id[order]
    boundary = np.any(sorted_cols[:, 1:] != sorted_cols[:, :-1], axis=0)
    twice = np.flatnonzero((ids[1:] == ids[:-1]) & ~boundary)
    if len(twice):
        at = int(twice[0]) + 1
        key = Key(*(t.vocab[f][sorted_cols[j, at]] for j, f in enumerate(GROUP_FIELDS)))
        raise DuplicateRows(f"{len(twice)} repeated episodes; first in group {key._asdict()}: "
                            f"episode fields {[t.vocab[f][t.codes[f][order[at]]] for f in EPISODE_FIELDS]} "
                            f"(an LLM platform must write `conversation` in every row)")
    starts = np.r_[0, np.flatnonzero(boundary) + 1]
    ends = np.r_[starts[1:], t.n]
    return {Key(*(t.vocab[f][sorted_cols[j, s]] for j, f in enumerate(GROUP_FIELDS))): order[s:e]
            for s, e in zip(starts, ends, strict=True)}


def load(results_dir, blocks: Sequence[str], metric_columns: Sequence[str], clean_margin: float | None = None) -> Table:
    """Read every committed row of the given blocks under `results_dir` (the directory that holds the block folders),
    from units/ and every worker shard (`core.committed_row_files`: a rows file without its done marker is a batch a crash
    cut before its done list or a unit still running or abandoned, and is counted in `uncommitted_files`, not read). A
    block with no units/ folder or no committed row is listed in `missing_blocks`. With `clean_margin` (runtime.recovery_margin) the attack rows are paired with their
    attack-free twin (LLM platform)."""
    builder, missing, torn, uncommitted = _Builder(metric_columns, clean_margin is not None), [], 0, 0
    for block in blocks:
        if not (Path(results_dir) / block / "units").is_dir():
            missing.append(block)
            continue
        files, skipped = core.committed_row_files(Path(results_dir) / block)
        uncommitted += skipped
        before = len(builder.attack_free)
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        torn += 1
                        continue
                    builder.add(raw, block)
        if len(builder.attack_free) == before:          # no committed row: a block without data, not an empty result
            missing.append(block)
    table = builder.finish(missing, torn, uncommitted)
    if clean_margin is not None:
        builder.pair_clean(table, clean_margin)
        table.empty_metrics = [m for m, v in table.metrics.items() if table.n == 0 or np.isnan(v).all()]
    return table
