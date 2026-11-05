"""MockPlatform: the core `Platform` contract over the torch engine.

`run_batch` cuts the units into chunks of `mock.batch_size` episodes (a performance setting: every episode has its own
keyed streams, so results do not depend on it), groups each chunk by horizon T (T = ceil(runtime.horizon_constant x the
graph's effective diameter), which differs between the graphs of one small-world or scale-free batch), simulates each
group as one tensor batch, turns the trace into rows with the metrics feature and appends them to the store. An episode
is atomic (milliseconds): a crash loses at most the group in progress, and a unit with saved rows but no done marker is
simulated again without writing its row twice.

Attack traces carry the paired attack-free accuracy: the same arm on the same episodes (graph, readings, sink, budget
cell, draws) without the attack. OPEN QUESTION (CONTRACT.md): the spec does not say whether the clean reference of
`recovery_time_accuracy` and `accuracy_clean` is the arm itself or no defence; `clean_reference_defender` is the one place
that decides it.

Torch runs on one thread inside the platform: the tensors are small, the sums are then ordered the same way on every
machine (results are bit-identical whatever the ambient thread count) and a busy machine cannot stall tiny kernels.
"""
from __future__ import annotations

import contextlib
import math
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch

from bazt_lab.core import public as core
from bazt_lab.features.metrics import public as metrics
from bazt_lab.features.mock_engine.private import engine, graphs
from bazt_lab.features.mock_engine.private import world as W
from bazt_lab.features.mock_engine.private.model import load_model

Tensor = torch.Tensor
CALIBRATION_MODES = ("open_loop", "closed_loop")
CLOSED_LOOP_TOLERANCE = 0.1       # closed loop stops when |realised deny rate - alpha| <= 10% of alpha (relative)
CLOSED_LOOP_PASSES = 40           # bisection passes at most; the closest level seen is returned if none is within tolerance
REQUIRED = ("id", "arm", "audit", "intervention", "topology", "attack", "agents", "seed")


@contextlib.contextmanager
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def clean_reference_defender(arm: core.ArmSpec) -> core.Defender:
    """The defender of the paired attack-free run: the arm itself, fresh (the utility the arm gives without an attack)."""
    return arm.factory()


def check_units(units: Sequence[Mapping], arm: core.ArmSpec) -> None:
    """Everything a batch must satisfy, checked before any work so that a refusal writes nothing."""
    W.check_homogeneous(units)
    for k, u in enumerate(units):
        missing = [f for f in REQUIRED if f not in u]
        if missing:
            raise ValueError(f"unit {k} has no {missing}; a mock unit carries {list(REQUIRED)}")
        engine.fraction(u, "audit")
        engine.fraction(u, "intervention")
    first = units[0]
    if arm.name != first["arm"]:
        raise ValueError(f"arm {arm.name!r} was given units of arm {first['arm']!r}")
    if "mock" not in arm.platforms:
        raise ValueError(f"arm {arm.name!r} does not run on the mock platform (platforms: {arm.platforms})")
    if first["topology"] not in graphs.TOPOLOGIES:
        raise ValueError(f"unknown topology {first['topology']!r}; the mock knows {graphs.TOPOLOGIES}")
    if first["attack"] not in W.ATTACKS:
        raise ValueError(f"unknown attack {first['attack']!r}; the mock knows {W.ATTACKS}")
    if len({repr(u.get("thresholds") or {}) for u in units}) > 1:
        raise ValueError("units of one batch must have the same thresholds")


def _batch_size(settings: Mapping) -> int:
    size = core.get(settings, "mock.batch_size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"mock.batch_size must be an integer >= 1, got {size!r}")
    return size


def _horizon_constant(settings: Mapping) -> float:
    c = core.get(settings, "runtime.horizon_constant")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not c > 0:
        raise ValueError(f"runtime.horizon_constant must be a number > 0, got {c!r}")
    return float(c)


def clean_accuracy(clean: core.EpisodeTrace, attackers: Tensor) -> Tensor:
    """float64 [B]: share of the attack run's victims (agents that are not insiders) correct at the last round of the
    paired attack-free run."""
    victims = ~attackers
    return ((clean.correct[:, -1] & victims).sum(1) / victims.sum(1)).to(torch.float64)


def threshold_at(values: Tensor, alpha: float) -> float:
    """Exact (1 - alpha) quantile: the k-th smallest of n values with k = ceil(n (1 - alpha)), so that at most alpha n values
    lie strictly above it. The product is rounded to 9 decimals first (0.98 x 100 is 98, not 98.00000000000001)."""
    n = values.numel()
    k = min(n, max(1, math.ceil(round(n * (1.0 - alpha), 9))))
    return float(torch.kthvalue(values, k).values)


def pool(pools: Dict[str, List[Tensor]], statistics: Mapping[str, Tensor], attempts: Tensor) -> None:
    """Add one round's decision statistics to the calibration pools. A [B, N, N] statistic contributes its value on each
    live delivery; a [B, N] statistic stands for the agent's live outgoing deliveries (the deny rate is a rate over
    deliveries), so it contributes once per delivery. NaN (not requested, not affordable) contributes nothing."""
    B, N, _ = attempts.shape
    for name, v in statistics.items():
        if not isinstance(v, torch.Tensor) or v.dtype != torch.float64 or tuple(v.shape) not in ((B, N, N), (B, N)):
            raise ValueError(f"Actions.statistics[{name!r}] must be a float64 tensor of shape {(B, N, N)} or {(B, N)}")
        if v.dim() == 3:
            values = v[attempts]
        else:
            sent, flat = attempts.sum(2).reshape(-1), v.reshape(-1)
            keep = ~torch.isnan(flat)
            values = flat[keep].repeat_interleave(sent[keep])
        pools.setdefault(name, []).append(values[~torch.isnan(values)])


def uniform_keys(rows: Sequence[Mapping]) -> List[dict]:
    """Rows that all carry the same keys (None where a row has none): a tracking table cannot gain columns later."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return [{k: r.get(k) for k in keys} for r in rows]


class MockPlatform:
    """Mock platform: a batch of episodes vectorised over B; the only Python loop is over rounds."""
    name = "mock"

    def __init__(self, rows_fn: Optional[Callable] = None) -> None:
        self._rows_fn = rows_fn or metrics.rows_from_trace

    # ------------------------------------------------------------------------------------------------ run_batch
    def run_batch(self, units: List[dict], arm: core.ArmSpec, settings: Mapping, store, tracker) -> List[dict]:
        units = list(units)
        check_units(units, arm)
        size, c = _batch_size(settings), _horizon_constant(settings)
        todo = [u for u in units if not store.is_done(u["id"])]
        rows: Dict[str, dict] = {}
        with one_thread():
            for start in range(0, len(todo), size):
                for group, trace in self._simulate_chunk(todo[start:start + size], arm, settings, c, with_clean=True):
                    made = self._rows(trace, settings)
                    self._save(store, group, made, rows)
                    tracker.log_rows("mock_units", uniform_keys(made))
        return [rows[u["id"]] for u in todo]

    # ------------------------------------------------------------------------------------------------ calibrate
    def calibrate(self, units: List[dict], arm: core.ArmSpec, settings: Mapping, store) -> Dict[str, float]:
        units = list(units)
        check_units(units, arm)
        if any(u["attack"] != "none" for u in units):
            raise ValueError("calibration runs are attack-free: every unit needs attack 'none'")
        if any(u["audit"] != "uncapped" or u["intervention"] != "uncapped" for u in units):
            raise ValueError("calibration runs the uncapped arm: audit and intervention must be 'uncapped'")
        alpha = core.get(settings, "budgets.alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
            raise ValueError(f"budgets.alpha must be a number in (0, 1), got {alpha!r}")
        mode = core.get(settings, "calibration.mode")
        if mode not in CALIBRATION_MODES:
            raise ValueError(f"calibration.mode must be one of {CALIBRATION_MODES}, got {mode!r}")
        size, c = _batch_size(settings), _horizon_constant(settings)
        pools: Dict[str, List[Tensor]] = {}
        collect = lambda statistics, attempts, active: pool(pools, statistics, attempts)      # noqa: E731
        with one_thread():
            for start in range(0, len(units), size):
                for group, trace in self._simulate_chunk(units[start:start + size], arm, settings, c, collect=collect):
                    if not pools:
                        raise ValueError(f"arm {arm.name!r} reported no Actions.statistics: there is nothing to calibrate")
                    fresh = [u for u in group if not store.is_done(u["id"])]
                    if fresh:
                        wanted = {u["id"] for u in fresh}
                        rows = self._rows(trace, settings)
                        self._save(store, fresh, [r for u, r in zip(group, rows, strict=True) if u["id"] in wanted], {})
            values = {}
            for name, parts in pools.items():
                values[name] = torch.cat(parts)
                if not values[name].numel():
                    raise ValueError(f"statistic {name!r} has no value to calibrate on (all NaN or no live delivery)")
            at = lambda level: {name: threshold_at(v, 1.0 - level) for name, v in values.items()}      # noqa: E731
            if mode == "open_loop":
                self.last_calibration = {"mode": mode, "passes": 1, "simulated": 1}
                return at(1.0 - float(alpha))
            return self._closed_loop(units, arm, settings, store, c, size, float(alpha), at)

    def _closed_loop(self, units, arm, settings, store, horizon_constant, size, alpha, at) -> Dict[str, float]:
        """Closed-loop calibration (CONTRACT.md): the arm acts with candidate thresholds {name: quantile_level(statistic)}
        on the attack-free episodes; bisect the common level until the realised deny rate is alpha within tolerance.
        Every pass is saved in the store (state key below), so a resumed calibration repeats none of them."""
        key = f"calibration-{arm.name}-{core.seed_of(alpha, *[u['id'] for u in units])}"
        saved = store.load_state(key) or {}
        cache = dict(saved.get("passes", {}))
        simulated = 1                                           # the statistics pass above
        worst: Dict[str, float] = {}                            # per pass: the largest deny rate of a single episode

        def rate_at(level: float) -> Tuple[float, Dict[str, float]]:
            nonlocal simulated
            th = at(level)
            hit = cache.get(repr(level))
            if hit is not None and hit["thresholds"] == th and "worst" in hit:
                worst[repr(level)] = hit["worst"]
                return hit["rate"], th
            denied = attempts = 0
            top = 0.0
            for start in range(0, len(units), size):
                for _, trace in self._simulate_chunk(units[start:start + size], arm, settings, horizon_constant, thresholds=th):
                    lost, sent = (trace.attempts & ~trace.delivered).sum((1, 2, 3)), trace.attempts.sum((1, 2, 3))
                    denied += int(lost.sum())
                    attempts += int(sent.sum())
                    top = max(top, float((lost / sent).max()))
            simulated += 1
            rate = denied / attempts
            cache[repr(level)] = {"rate": rate, "thresholds": th, "worst": top}
            worst[repr(level)] = top
            store.save_state(key, {"passes": cache})
            return rate, th

        lo, hi, level, seen = 0.0, 1.0, 1.0 - alpha, []
        for n in range(1, CLOSED_LOOP_PASSES + 1):  # noqa: B007
            rate, th = rate_at(level)
            seen.append((level, rate, th))
            if abs(rate - alpha) <= CLOSED_LOOP_TOLERANCE * alpha or hi - lo < 1e-12:
                break
            lo, hi = (level, hi) if rate > alpha else (lo, level)          # a higher level means a higher threshold: fewer denies
            level = 0.5 * (lo + hi)
        # The realised rate can jump over the tolerance band (one more denial raises the next statistics): then the level
        # with the highest rate not above the band is returned, never one that denies more than alpha (1 + tolerance).
        ceiling = alpha * (1.0 + CLOSED_LOOP_TOLERANCE)
        below = [x for x in seen if x[1] <= ceiling]
        best = max(below, key=lambda x: x[1]) if below else min(seen, key=lambda x: x[1])
        self.last_calibration = {"mode": "closed_loop", "level": best[0], "rate": best[1], "passes": n, "simulated": simulated,
                                 "within_tolerance": abs(best[1] - alpha) <= CLOSED_LOOP_TOLERANCE * alpha,
                                 "worst_episode_rate": worst[repr(best[0])]}
        return best[2]

    # --------------------------------------------------------------------------------------------------- engine
    def _rows(self, trace: core.EpisodeTrace, settings: Mapping) -> List[dict]:
        rows = self._rows_fn(trace, settings)
        if len(rows) != len(trace.unit_ids):
            raise RuntimeError(f"the rows function returned {len(rows)} rows for {len(trace.unit_ids)} episodes")
        return rows

    @staticmethod
    def _save(store, group: Sequence[Mapping], rows: Sequence[dict], into: Dict[str, dict]) -> None:
        """Save a group's rows and mark its units done: one batch file and one done list when the store has the batch API,
        else one file per unit. A unit that an interrupted run already saved rows for (per-unit files) is only marked done."""
        batch = {}
        for u, row in zip(group, rows, strict=True):
            into[u["id"]] = row
            if store.resume_rows(u["id"]):
                store.mark_done(u["id"])
            elif hasattr(store, "append_batch"):
                batch[u["id"]] = [row]
            else:
                store.append_rows(u["id"], [row])
                store.mark_done(u["id"])
        if batch:
            store.append_batch(batch)

    def _simulate_chunk(self, chunk: Sequence[Mapping], arm: core.ArmSpec, settings: Mapping, horizon_constant: float,
                        with_clean: bool = False, collect: Optional[Callable] = None,
                        thresholds: Optional[Mapping[str, float]] = None) -> Iterator[Tuple[List[Mapping], core.EpisodeTrace]]:
        """Simulate a chunk of one arm / topology / attack / N: one (units, trace) per distinct horizon, in input order within
        the group."""
        first = chunk[0]
        topology, N, family = first["topology"], first["agents"], first["attack"]
        model = load_model(settings)
        adj = W.make_graphs(chunk, settings, topology, N)
        horizon = graphs.horizon_rounds(graphs.diameters(adj), horizon_constant)
        if thresholds is None:
            thresholds = {} if collect is not None else dict(first.get("thresholds") or {})
        for T in sorted(set(horizon.tolist())):
            rows = (horizon == T).nonzero().flatten()
            units, adj_t = [chunk[i] for i in rows.tolist()], adj[rows]
            draws = W.draw_noise(units, topology, N, T)

            def run(fam: str, defender: core.Defender, units=units, adj_t=adj_t, T=T, draws=draws) -> core.EpisodeTrace:
                world = W.build_world(units, settings, topology, N, fam, adj_t, T, draws)
                ctx = engine.make_context(units, world, arm.name, settings, thresholds)
                return engine.simulate(world, model, defender, ctx, arm.needs_truth, collect, units)[0]
            trace = run(family, arm.factory())
            if with_clean and family != "none":
                trace.meta["clean_accuracy"] = clean_accuracy(run("none", clean_reference_defender(arm)), trace.attackers)
            yield units, trace
