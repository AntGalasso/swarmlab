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
        sink = u.get("sink")
        if sink is not None and (isinstance(sink, bool) or not isinstance(sink, int) or not 0 <= sink < u["agents"]):
            raise ValueError(f"unit {u['id']}: sink must be an agent index in [0, {u['agents']}) or None, got {sink!r}")
        W._params(u)                                   # attack_params: refused here, not in a later chunk
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


def closed_loop_params(settings: Mapping) -> Tuple[float, int]:
    """(tolerance, max passes) of the closed loop: owner/coordinator settings calibration.tolerance and calibration.max_passes."""
    tol, passes = core.get(settings, "calibration.tolerance"), core.get(settings, "calibration.max_passes")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 0 < tol < 1:
        raise ValueError(f"calibration.tolerance must be a number in (0, 1), got {tol!r}")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise ValueError(f"calibration.max_passes must be an integer >= 1, got {passes!r}")
    return float(tol), passes


def _horizon_constant(settings: Mapping) -> float:
    c = core.get(settings, "runtime.horizon_constant")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not c > 0:
        raise ValueError(f"runtime.horizon_constant must be a number > 0, got {c!r}")
    return float(c)


def clean_accuracy(clean: core.EpisodeTrace, attackers: Tensor) -> Tensor:
    """float64 [B]: share of the attack run's victims (agents that are not insiders) correct at the last round of the
    paired attack-free run."""
    victims = ~attackers
    return (clean.correct[:, -1] & victims).sum(1).to(torch.float64) / victims.sum(1).to(torch.float64)


TWIN_SCHEDULE = "twin_timing_schedule"


def clean_schedule(unit: Mapping) -> dict:
    """The unit as its attack-free run sees it: a matched-timing unit acts at the rounds of the grid's attack-free unit
    (BAZT's record of that unit, D116), which the runner hands over as `twin_timing_schedule`."""
    clean = {k: v for k, v in unit.items() if k not in ("timing_schedule", TWIN_SCHEDULE)}
    if TWIN_SCHEDULE in unit:
        clean["timing_schedule"] = unit[TWIN_SCHEDULE]
    return clean


def check_clean_schedules(units: Sequence[Mapping]) -> None:
    """A unit with a timing schedule whose attack-free run must be simulated needs the schedule of that run."""
    lacking = [u["id"] for u in units if "timing_schedule" in u and TWIN_SCHEDULE not in u]
    if lacking:
        raise ValueError(f"unit {lacking[0]!r} carries timing_schedule but no {TWIN_SCHEDULE}: its attack-free run acts at "
                         f"the rounds of the attack-free unit (D116), which must be handed over")


def twin_unit(unit: Mapping) -> dict:
    """The attack-free twin of an attacked unit (D109): the same unit with attack 'none' and the twin's id, its
    attack_params cut to the keys an attack-free run reads (W.ATTACK_FREE_PARAMS: the adaptive auditor's detection rate),
    and its own timing schedule (`clean_schedule`). That is the grid's own attack-free unit: the variant's auditor_tpr
    stays, the attacker's strategy goes."""
    twin = {k: v for k, v in clean_schedule(unit).items() if k != "twin_id"}
    twin.update(id=unit["twin_id"], attack="none")
    if isinstance(unit.get("attack_params"), Mapping):
        twin["attack_params"] = {k: v for k, v in unit["attack_params"].items() if k in W.ATTACK_FREE_PARAMS}
    return twin


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
        if not isinstance(v, torch.Tensor) or v.dtype not in (torch.float64, torch.float32) or tuple(v.shape) not in ((B, N, N), (B, N)):
            raise ValueError(f"Actions.statistics[{name!r}] must be a float tensor (ctx.dtype) of shape {(B, N, N)} or {(B, N)}")
        stat = v.to(device="cpu", dtype=torch.float64)             # thresholds are float64 whatever the compute dtype
        attempts = attempts.to("cpu")
        if stat.dim() == 3:
            values = stat[attempts]
        else:
            sent, flat = attempts.sum(2).reshape(-1), stat.reshape(-1)
            keep = ~torch.isnan(flat)
            values = flat[keep].repeat_interleave(sent[keep])
        pools.setdefault(name, []).append(values[~torch.isnan(values)])


def pick_pass(seen: Sequence[tuple], alpha: float, tolerance: float) -> tuple:
    """D112, the same rule as the LLM platform: of the closed-loop passes (level, rate, thresholds), the one with the
    highest rate not above alpha (1 + tolerance), the first on a tie; when every pass is above it, the lowest rate. A pass
    within the band is that pass (the search stops at it, and every earlier pass not above the band was below it)."""
    below = [x for x in seen if x[1] <= alpha * (1.0 + tolerance)]
    return max(below, key=lambda x: x[1]) if below else min(seen, key=lambda x: x[1])


def uniform_keys(rows: Sequence[Mapping]) -> List[dict]:
    """Rows that all carry the same keys (None where a row has none): a tracking table cannot gain columns later."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return [{k: r.get(k) for k in keys} for r in rows]


class MockPlatform:
    """Mock platform: a batch of episodes vectorised over B; the only Python loop is over rounds."""
    name = "mock"

    @property
    def adaptive_conditions(self) -> tuple:
        """Every condition the adaptive feature searches: its best responses reach the engine as attack_params."""
        from bazt_lab.features.adaptive import public as adaptive
        return tuple(adaptive.CONDITIONS)

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
                chunk = [u for u in todo[start:start + size] if not store.is_done(u["id"])]     # a twin may have been saved meanwhile
                if not chunk:
                    continue
                for group, trace, twins in self._simulate_chunk(chunk, arm, settings, c, with_clean=True, store=store):
                    made = self._rows(trace, settings)
                    self._save(store, group, made, rows)
                    if twins is not None:                      # the twin units' own results, never simulated twice
                        fresh = [(t, r) for t, r in zip(twins[0], self._rows(twins[1], settings), strict=True)
                                 if not store.is_done(t["id"])]
                        self._save(store, [t for t, _ in fresh], [r for _, r in fresh], {})
                    tracker.log_rows("mock_units", uniform_keys(made))
        return [rows[u["id"]] for u in todo if u["id"] in rows]

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
                for group, trace, _ in self._simulate_chunk(units[start:start + size], arm, settings, c, collect=collect):
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
                self.last_calibration = {"mode": mode, "passes": 1, "simulated": 1, "alpha": float(alpha), "converged": True}
                return at(1.0 - float(alpha))
            return self._closed_loop(units, arm, settings, store, c, size, float(alpha), at)

    def realised_rate(self, units: Sequence[Mapping], arm: core.ArmSpec, settings: Mapping,
                      thresholds: Mapping[str, float]) -> Tuple[float, float]:
        """(deny rate, largest single-episode deny rate) of the arm acting with `thresholds` on these units: the quantity the
        closed loop drives to alpha. D108: when the arm reports Actions.threshold_restricted only those deliveries count
        (threshold_restricted & attempts & ~delivered), else every restricted or lost delivery (attempts & ~delivered)."""
        size, c = _batch_size(settings), _horizon_constant(settings)
        counted_total = sent_total = 0
        worst = 0.0
        with one_thread():
            for start in range(0, len(units), size):
                extras: dict = {}
                for _, trace, _ in self._simulate_chunk(list(units)[start:start + size], arm, settings, c,
                                                     thresholds=thresholds, extras=extras):
                    lost = trace.attempts & ~trace.delivered           # restricted or lost (isolation included)
                    counted = extras["threshold_restricted"] & lost if "threshold_restricted" in extras else lost
                    counted, sent = counted.sum((1, 2, 3)), trace.attempts.sum((1, 2, 3))
                    counted_total += int(counted.sum())
                    sent_total += int(sent.sum())
                    worst = max(worst, float((counted / sent).max()))
        return counted_total / sent_total, worst

    def _closed_loop(self, units, arm, settings, store, horizon_constant, size, alpha, at) -> Dict[str, float]:
        """Closed-loop calibration (CONTRACT.md): the arm acts with candidate thresholds {name: quantile_level(statistic)}
        on the attack-free episodes; bisect the common level until the realised deny rate is alpha within tolerance.
        Every pass is saved in the store (state key below), so a resumed calibration repeats none of them."""
        tolerance, max_passes = closed_loop_params(settings)
        key = f"calibration2-{arm.name}-{core.seed_of(alpha, *[u['id'] for u in units])}"
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
            rate, top = self.realised_rate(units, arm, settings, th)
            simulated += 1
            cache[repr(level)] = {"rate": rate, "thresholds": th, "worst": top}
            worst[repr(level)] = top
            store.save_state(key, {"passes": cache})
            return rate, th

        lo, hi, level, seen = 0.0, 1.0, 1.0 - alpha, []
        for n in range(1, max_passes + 1):  # noqa: B007
            rate, th = rate_at(level)
            seen.append((level, rate, th))
            if abs(rate - alpha) <= tolerance * alpha or hi - lo < 1e-12:
                break
            lo, hi = (level, hi) if rate > alpha else (lo, level)          # a higher level means a higher threshold: fewer denies
            level = 0.5 * (lo + hi)
        # Not converged (the rate moves in steps over the band, or alpha is unreachable): D112, the highest pass not above
        # alpha (1 + tolerance), as on the LLM platform; `converged` (alias within_tolerance) says whether it is in the band.
        best = pick_pass(seen, alpha, tolerance)
        converged = abs(best[1] - alpha) <= tolerance * alpha
        self.last_calibration = {"mode": "closed_loop", "level": best[0], "rate": best[1], "alpha": alpha, "passes": n,
                                 "simulated": simulated, "converged": converged, "within_tolerance": converged,
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
        if hasattr(store, "append_batch"):                  # one file, one fsync, one done list; no per-unit file system calls
            for u, row in zip(group, rows, strict=True):
                into[u["id"]] = row
            store.append_batch({u["id"]: [row] for u, row in zip(group, rows, strict=True)})
            return
        for u, row in zip(group, rows, strict=True):
            into[u["id"]] = row
            if store.resume_rows(u["id"]):
                store.mark_done(u["id"])
            else:
                store.append_rows(u["id"], [row])
                store.mark_done(u["id"])

    def _simulate_chunk(self, chunk: Sequence[Mapping], arm: core.ArmSpec, settings: Mapping, horizon_constant: float,
                        with_clean: bool = False, collect: Optional[Callable] = None,
                        thresholds: Optional[Mapping[str, float]] = None, extras: Optional[dict] = None, store=None) -> Iterator[Tuple[List[Mapping], core.EpisodeTrace]]:
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

            def run(fam: str, defender: core.Defender, us=units, adj_t=adj_t, T=T, draws=draws) -> core.EpisodeTrace:
                world = W.build_world(us, settings, topology, N, fam, adj_t, T, draws)
                ctx = engine.make_context(us, world, arm.name, settings, thresholds)
                return engine.simulate(world, model, defender, ctx, arm.needs_truth, collect, us, extras if fam == family else None)[0]
            trace = run(family, arm.factory())
            twins = None
            if with_clean and family != "none":
                twin_units = [twin_unit(u) for u in units] if all("twin_id" in u for u in units) else None
                reference = self._stored_reference(store, twin_units) if twin_units and family not in W.INSIDERS else None
                if reference is not None:                      # D109: the twin's own finished run
                    trace.meta["clean_accuracy"] = reference
                elif twin_units is not None:                   # D109: simulate the twin keyed by its own id, save it as its result
                    check_clean_schedules(units)
                    clean = run("none", clean_reference_defender(arm), twin_units)
                    trace.meta["clean_accuracy"] = clean_accuracy(clean, trace.attackers)
                    twins = (twin_units, clean)
                else:
                    check_clean_schedules(units)
                    clean = run("none", clean_reference_defender(arm), [clean_schedule(u) for u in units])
                    trace.meta["clean_accuracy"] = clean_accuracy(clean, trace.attackers)
            yield units, trace, twins

    @staticmethod
    def _stored_reference(store, twin_units: Sequence[Mapping]) -> Optional[Tensor]:
        """float64 [B]: the finished twins' accuracy_clean, when every twin of the group is done and the store can give its row."""
        if store is None or not hasattr(store, "rows_of") or not all(store.is_done(t["id"]) for t in twin_units):
            return None
        rows = [store.rows_of(t["id"]) for t in twin_units]
        if any(len(r) != 1 or not isinstance(r[0].get("accuracy_clean"), (int, float)) for r in rows):
            return None
        return torch.tensor([r[0]["accuracy_clean"] for r in rows], dtype=torch.float64)
