"""The exact best-response search of the adaptive attacker (EXPERIMENT_PLAN.md section 3).

For one cell (arm, budget cell, topology, family, N, condition) the attacker's strategy space is the finite list of
`attack_params` of `space.candidates`. Every candidate is simulated on the same search episodes (common random
numbers: the episodes depend on (topology, N, seed) only, never on the candidate) and the strategy with the highest
mean final-round ASR wins. Ties go to the earlier candidate, so a strategy beats the family's default attack
(candidate 0, `{}`) only by being strictly better.

The knowledge level of the condition sets which episodes a strategy is chosen on:
  cell     one strategy for the cell, chosen on the search seeds (never evaluated), the way a gray-box attacker that
           does not see the defender's current allocation must commit;
  seed     one strategy per evaluated seed, chosen on that seed's own episodes (the allocation of the seed is known);
  episode  one strategy per evaluated (seed, sink) episode, chosen on that episode.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from bazt_lab.features.adaptive.private import space as sp
from bazt_lab.features.adaptive.private import table as tb
from bazt_lab.features.runner import public as runner

REQUIRED = ("arm", "attack", "agents", "condition")
BLOCK = "mock_adaptive_search"
METRIC = "asr_final"            # the safety endpoint of the confirmatory hypotheses (D63), a column of the metrics rows


@dataclass(frozen=True)
class Job:
    """The search of one cell. `pairs` are the (seed, sink) episodes simulated for every candidate; `groups` maps each
    table key (seed, sink; None where the level is coarser) to the indices of the pairs its strategy is chosen on."""
    cell: dict
    level: str
    pairs: Tuple[tuple, ...]
    groups: Dict[tuple, Tuple[int, ...]]
    ids: Mapping[tuple, str]            # (seed, sink) -> the episode id every candidate of that episode runs under

    @property
    def arm(self) -> str:
        return self.cell["arm"]

    @property
    def family(self) -> str:
        return self.cell["attack"]

    @property
    def agents(self) -> int:
        return self.cell["agents"]

    @property
    def condition(self) -> str:
        return self.cell["condition"]


def _episode_id(cell: dict, seed, sink) -> str:
    """The id of a search episode. Every candidate of one episode runs under it, so the platform's arm-keyed streams
    (random allocation, tie-breaks) are the same for all of them: common random numbers across candidates."""
    digest = hashlib.sha256(tb.cell_id(cell).encode()).hexdigest()[:12]
    return f"{cell['arm']}-search-{digest}-{seed}-{sink}"


def _job(cell: dict, eval_units: Mapping[tuple, Mapping], space: sp.Space) -> Job:
    """eval_units: (seed, sink) -> an evaluation unit of the cell. A search on the evaluation episodes themselves runs
    under the evaluation unit's id (its `id` field, else `runner.unit_id`), so the searched ASR of the chosen strategy
    is exactly the ASR the evaluation will measure."""
    cond = cell["condition"]
    if cond not in space.knowledge:
        raise sp.Refusal(f"condition {cond!r} is not in mock.adaptive.conditions {list(space.knowledge)}")
    level = space.knowledge[cond]
    if level == "cell":
        lo, hi = space.search_seeds
        clash = sorted({s for s, _ in eval_units} & set(range(lo, hi + 1)))
        if clash:
            raise sp.Refusal(f"{tb.cell_id(cell)}: evaluation seeds {clash} are among the search seeds {[lo, hi]}: "
                             f"the attacker must be optimised on seeds that are not evaluated")
        sinks = range(cell["agents"]) if space.every_sink else (None,)
        pairs = tuple((s, k) for s in range(lo, hi + 1) for k in sinks)
        return Job(cell, level, pairs, {(None, None): tuple(range(len(pairs)))},
                   {p: _episode_id(cell, *p) for p in pairs})
    pairs = tuple(eval_units)
    groups: Dict[tuple, List[int]] = {}
    for i, (seed, sink) in enumerate(pairs):
        groups.setdefault((seed, None) if level == "seed" else (seed, sink), []).append(i)
    return Job(cell, level, pairs, {k: tuple(v) for k, v in groups.items()},
               {p: eval_units[p].get("id") or runner.unit_id(eval_units[p]) for p in pairs})


def plan_jobs(units: Iterable[Mapping], space: sp.Space) -> List[Job]:
    """One job per cell of the evaluation units, in first-seen order; attack-free units have nothing to search.
    Refuses before anything runs: a unit without a required field, an unknown condition, too many candidates."""
    cells: Dict[str, tuple] = {}
    for u in units:
        for field in REQUIRED:
            if field not in u:
                raise sp.Refusal(f"evaluation unit has no field {field!r}: {dict(u)}")
        if u["attack"] == tb.ATTACK_FREE:
            continue
        cell = tb.cell_of(u)
        cells.setdefault(tb.cell_id(cell), (cell, {}))[1].setdefault((u["seed"], u.get("sink")), u)
    jobs = [_job(cell, pairs, space) for cell, pairs in cells.values()]
    for job in jobs:
        sp.checked_count(job.family, job.agents, job.condition, space)
    return jobs


def search_episodes(units: Iterable[Mapping], space: sp.Space) -> int:
    """Episodes the search simulates for these evaluation units (candidates x search pairs, summed over cells)."""
    return sum(sp.checked_count(j.family, j.agents, j.condition, space) * len(j.pairs)
               for j in plan_jobs(units, space))


# ================================================================================================ simulation
class _Scratch:
    """The store handed to the platform for one chunk: rows live in memory, because the search checkpoints its own
    compact progress (one line per chunk) instead of two files per episode."""

    def __init__(self) -> None:
        self.rows: Dict[str, list] = {}
        self.done: set = set()
        self.states: Dict[str, dict] = {}

    def is_done(self, uid):
        return uid in self.done

    def append_rows(self, uid, rows):
        self.rows.setdefault(uid, []).extend(rows)

    def resume_rows(self, uid):
        return list(self.rows.get(uid, []))

    def mark_done(self, uid):
        self.done.add(uid)

    def save_state(self, uid, state):
        self.states[uid] = dict(state)

    def load_state(self, uid):
        return self.states.get(uid)


def run_chunk(platform, arm_spec, settings, tracker, batch: List[dict]) -> List[float]:
    """Final-round ASR of each unit of the batch, in order. The platform must return one row per unit, in unit order."""
    rows = platform.run_batch(batch, arm_spec, settings, _Scratch(), tracker)
    if len(rows) != len(batch):
        raise RuntimeError(f"the platform returned {len(rows)} rows for {len(batch)} units")
    out = []
    for unit, row in zip(batch, rows, strict=True):
        named = row.get("id", row.get("unit_id", unit["id"]))       # metrics rows carry `id` (unit dict) or `unit_id`
        if named != unit["id"]:
            raise RuntimeError(f"the platform returned rows in another order than the units ({named} for {unit['id']})")
        value = row.get(METRIC, math.nan)
        if not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"unit {unit['id']}: the platform's row has no usable {METRIC} (got {value!r})")
        out.append(float(value))
    return out


def _load_progress(path: Path, vals: torch.Tensor) -> None:
    """Fill `vals` (flat, NaN = pending) from the progress lines. A torn last line is dropped from the file."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    good = []
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            break
        f, a = rec["f"], rec["asr"]
        if len(f) != len(a) or any(not 0 <= i < vals.numel() for i in f):
            raise ValueError(f"{path}: progress does not fit this plan ({vals.numel()} episodes); delete it")
        vals[torch.tensor(f, dtype=torch.int64)] = torch.tensor(a, dtype=torch.float64)
        good.append(line)
    if len(good) != len(lines):
        path.write_text("".join(line + "\n" for line in good), encoding="utf-8")


def _append_progress(path: Path, indices: Sequence[int], asr: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"f": list(indices), "asr": list(asr)}, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def chunks_of(pending: Sequence[int], P: int, size: int) -> List[List[int]]:
    """Cut the pending episodes (flat index f = c * P + p) into chunks of at most `size` with no pair p twice: the
    candidates of one episode share an id, and a platform batch needs distinct ids."""
    out: List[List[int]] = []
    cur: List[int] = []
    used: set = set()
    for f in pending:
        p = f % P
        if p in used or len(cur) == size:
            out.append(cur)
            cur, used = [], set()
        cur.append(f)
        used.add(p)
    if cur:
        out.append(cur)
    return out


def evaluate(job: Job, cands: List[dict], run: Callable[[List[dict]], List[float]], progress: Path, batch_size: int,
             on_chunk: Callable[[int], None], thresholds: Mapping) -> torch.Tensor:
    """ASR of every (candidate, search pair) as float64 [C, P]. Episode f = c * P + p; chunks of pending episodes (at
    most `batch_size`, no episode twice) go to `run` and are appended to `progress` as they finish, so a rerun resumes
    after the last saved chunk. Nothing pending runs nothing."""
    C, P = len(cands), len(job.pairs)
    vals = torch.full((C * P,), math.nan, dtype=torch.float64)
    _load_progress(progress, vals)
    pending = torch.nonzero(torch.isnan(vals)).flatten().tolist()
    done = C * P - len(pending)
    template = {**job.cell, "block": BLOCK}
    for indices in chunks_of(pending, P, batch_size):
        batch = []
        for f in indices:
            c, p = divmod(f, P)
            seed, sink = job.pairs[p]
            batch.append({**template, "seed": seed, "sink": sink, "attack_params": cands[c],
                          "id": job.ids[(seed, sink)], "thresholds": thresholds})
        asr = run(batch)
        vals[torch.tensor(indices, dtype=torch.int64)] = torch.tensor(asr, dtype=torch.float64)
        _append_progress(progress, indices, asr)
        done += len(indices)
        on_chunk(done)
    return vals.view(C, P)


def victims_of(family: str, agents: int, n_insiders: int) -> int:
    """Agents whose answer counts in ASR: everyone but the insiders (metrics: victims = agents that are not attackers)."""
    return agents - (n_insiders if family in sp.INSIDER_FAMILIES else 0)


def _means(rows: List[List[float]], n_victims: int) -> Tuple[List[float], list]:
    """(mean ASR, comparable total) of each candidate over its episodes. ASR is (wrong agents) / n_victims, so when every value is such a
    multiple the integer totals are compared and equal totals tie exactly; otherwise the exactly rounded fsum."""
    snapped = [[round(v * n_victims) for v in row] for row in rows]
    if n_victims > 0 and all(abs(v * n_victims - k) < 1e-6 for row, ks in zip(rows, snapped, strict=True)
                             for v, k in zip(row, ks, strict=True)):
        return [sum(ks) / (n_victims * len(ks)) if len(ks) > 1 else row[0] for row, ks in zip(rows, snapped, strict=True)],             [sum(ks) for ks in snapped]
    return [row[0] if len(row) == 1 else math.fsum(row) / len(row) for row in rows],         [row[0] if len(row) == 1 else math.fsum(row) for row in rows]


def choose(job: Job, cands: List[dict], vals: torch.Tensor, n_victims: int = 0) -> List[dict]:
    """The best candidate of each table key: highest mean ASR over the key's episodes, the earlier candidate on a tie.
    Totals are compared in exact arithmetic (see `_means`)."""
    records = []
    for (seed, sink), idx in job.groups.items():
        means, totals = _means(vals[:, list(idx)].tolist(), n_victims)
        best = max(range(len(cands)), key=totals.__getitem__)          # the first maximum: earlier candidate wins
        records.append(dict(seed=seed, sink=sink, attack_params=cands[best], asr=means[best], baseline_asr=means[0],
                            candidates=len(cands), episodes=len(idx)))
    return records
