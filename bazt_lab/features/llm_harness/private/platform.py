"""LLMPlatform: the Platform contract over G-Safeguard (spec 4.1). One unit = one seed run (private/seedrun.py).

Constructing it does nothing (no network, no secret, no file): the patched G-Safeguard copy, the backend and the judge
are built when the first unit runs. `llm.api = "fake"` selects the offline FakeBackend + FakeJudge + synthetic items
(tests, scripts/e2e_llm_fake.py); anything else is the live path (adapters/live.py), which is behind the kickoff guard.
"""
from __future__ import annotations

import contextlib
import copy
import sys
from typing import Callable, Dict, List, Mapping

import torch

from bazt_lab.core import public as core
from bazt_lab.features.gnn import public as gnn
from bazt_lab.features.judge import public as judge_feature
from bazt_lab.features.llm_data import public as data
from bazt_lab.features.llm_harness.private import fake, prompts, schedule
from bazt_lab.features.llm_harness.private.seedrun import Env, SeedRun


class _NoDefence:
    name = "no_defence"

    def reset(self, ctx) -> None:
        pass

    def decide(self, obs) -> core.Actions:
        return core.Actions()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        pass


CALIBRATION_MODES = ("open_loop", "closed_loop")
NO_DEFENCE = core.ArmSpec("no_defence", _NoDefence, platforms=("llm",))


class _LazyJudge:
    """A judge built on its first score() call, so an arm that never audits needs no judge, provider or key."""

    def __init__(self, build: Callable[[], object]) -> None:
        self._build, self._judge = build, None

    @property
    def stats(self):
        return getattr(self._judge, "stats", None)

    def score(self, texts):
        if self._judge is None:
            self._judge = self._build()
        return self._judge.score(texts)

    def close(self) -> None:
        close = getattr(self._judge, "close", None)         # nothing was built when no arm audited
        if callable(close):
            close()


def _close_all(owned: List[object]) -> None:
    """Close what a call built (a live backend's cache connection, a judge's HTTP client and cache), every one even
    when an earlier close fails; the first failure is raised after all were tried, and never hides the call's own error."""
    first = None
    for obj in owned:
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:                                    # keep closing the rest
                first = first or e
    if first is not None and sys.exc_info()[0] is None:
        raise first


def _fake_judge():
    return judge_feature.FakeJudge({prompts.REPLICATION_MARKER: 0.9, fake.AITM_MARK: 0.9, "the user is right": 0.9,
                                    "Action: AttackTool": 0.9})


class LLMPlatform:
    name = "llm"
    # D110: the held-out check replays the mock's gray-box best response (private/schedule.py); SeedRun refuses a unit
    # that carries another condition
    adaptive_conditions: tuple = schedule.ADAPTIVE_CONDITIONS

    def __init__(self, backend=None, judge=None, sample=None, families=None) -> None:
        """Every argument replaces what settings would select (tests): a backend, a Judge, a sampler
        `(task, seed, k, settings) -> [TaskItem]`, a loader `family -> modules`."""
        self._given = (backend, judge, sample, families)
        self._families: Dict[str, object] = {}
        self._root = None

    # ----------------------------------------------------------------------------------------- environment
    def _load_family(self, name: str):
        from bazt_lab.features.llm_harness.adapters import upstream  # git apply of the patch set, on first use
        if self._root is None:
            self._root = upstream.prepare()
        if name not in self._families:
            self._families[name] = upstream.load_family(self._root, name)
        return self._families[name]

    def _environment(self, settings: Mapping, store, owned: List[object] | None = None) -> Env:
        """`owned` collects what this call builds itself (a given backend or judge belongs to the caller), to be closed."""
        backend, judge, sample, families = self._given
        if core.is_set(settings, "llm.api") and core.get(settings, "llm.api") == "fake":
            backend, judge, sample = backend or fake.FakeBackend(), judge or _fake_judge(), sample or fake.sample
        else:
            from bazt_lab.features.llm_harness.adapters import live
            allow = bool(core.get(settings, "llm.allow_live"))
            shared = store.root.parent                # one cache for all blocks of a profile: arms share requests
            if backend is None:
                backend = live.backend(settings, allow_live=allow, cache_path=shared / "llm_cache.sqlite")
                if owned is not None:
                    owned.append(backend)
            if judge is None:
                judge = _LazyJudge(lambda: judge_feature.live_judge(
                    settings, allow_live=allow, cache_path=shared / "judge_cache.sqlite"))
                if owned is not None:
                    owned.append(judge)
            sample = sample or data.sample
        return Env(backend, judge, sample, families or self._load_family)

    # ---------------------------------------------------------------------------------------- the contract
    def run_batch(self, units: List[dict], arm: core.ArmSpec, settings: Mapping, store, tracker) -> List[dict]:
        owned: List[object] = []
        try:
            env = self._environment(settings, store, owned)
            rows: List[dict] = []
            for unit in units:
                if not store.is_done(unit["id"]):
                    if "edge_probability" in unit:                                  # a gnn_train unit
                        self._train(unit, settings, env, store)
                    else:
                        SeedRun(unit, arm, settings, env, store, calibrating=False).run()
                rows += store.resume_rows(unit["id"])
            # A unit is marked done once its conversations are logged, so a crash while the GNN trains is repaired here: on
            # every run, each family whose logs are complete and whose training has not finished is trained.
            for unit in (u for u in units if "edge_probability" in u):
                family = f"{unit['attack']}__{unit['task']}"
                if not gnn.training_finished(settings, family):
                    with contextlib.suppress(gnn.TrainingDataIncomplete):
                        gnn.train_unit(dict(unit), settings)
            return rows
        finally:
            _close_all(owned)

    # ------------------------------------------------------------------------------------------ calibration
    def _pass(self, units, arm, settings, env, store, thresholds: Mapping, tag: str):
        """One calibration pass over all units, the arm acting on `thresholds`: (pooled statistics, realised deny rate)."""
        pooled: Dict[str, List[torch.Tensor]] = {}
        acted = total = 0
        for unit in units:
            run = SeedRun(unit, arm, settings, env, store, calibrating=True, thresholds=thresholds, tag=tag)
            for name, values in run.run().items():
                pooled.setdefault(name, []).append(values)
            acted, total = acted + run.acted, total + run.total
        return {k: torch.cat(v) for k, v in pooled.items()}, (acted / total if total else 0.0)

    def calibrate(self, units: List[dict], arm: core.ArmSpec, settings: Mapping, store) -> Dict[str, float]:
        """Attack-free conversations of the uncapped arm; per statistic the threshold at deny rate alpha (NaN ignored;
        `[1,N,N]` statistics over open deliveries, `[1,N]` over active agents). `calibration.mode` (owner key, no default):

        open_loop    the (1 - alpha) quantile of the statistics of one pass in which the arm acts on nothing.
        closed_loop  the arm acts on attack-free conversations, with the same search as mock_engine: candidate thresholds are
                     the quantiles at one common level of the open-loop statistics, the level is bisected (start 1 - alpha,
                     at most `calibration.max_passes` passes, deterministic) until the realised rate is alpha within
                     `calibration.tolerance` x alpha (both owner-set settings, no default). The realised rate counts the deliveries restricted because a calibrated
                     statistic crossed its threshold (`Actions.threshold_restricted`, D108); an arm that does not provide
                     it counts every restriction. If no pass lands within the band (one more denial raises the next
                     statistics, so the rate can jump over it) the pass with the highest rate not above alpha x (1 +
                     tolerance) is returned, never one that denies more; if every pass is above it, the lowest rate.

        `self.last_calibration` = {mode, level, rate, passes, simulated, within_tolerance, alpha}. Callable again on
        finished units: every pass is read back from the store."""
        mode = core.get(settings, "calibration.mode")
        if mode not in CALIBRATION_MODES:
            raise ValueError(f"calibration.mode must be one of {CALIBRATION_MODES}, got {mode!r}")
        if mode == "closed_loop":                          # owner-set, no default: refuse before any pass is paid for
            core.get(settings, "calibration.tolerance")
            core.get(settings, "calibration.max_passes")
        for unit in units:
            if unit["attack"] != "none":
                raise ValueError(f"calibration runs attack-free conversations, unit {unit['id']} has attack {unit['attack']!r}")
        owned: List[object] = []
        try:
            return self._calibrate(units, arm, settings, store, mode, self._environment(settings, store, owned))
        finally:
            _close_all(owned)

    def _calibrate(self, units, arm, settings, store, mode: str, env: Env) -> Dict[str, float]:
        alpha = float(core.get(settings, "budgets.alpha"))
        base, rate = self._pass(units, arm, settings, env, store, {}, "")
        if not base:
            raise ValueError(f"arm {arm.name} reported no statistic during calibration")
        for name, values in base.items():
            if values.numel() == 0:
                raise ValueError(f"statistic {name!r} has no finite value in the calibration conversations")

        def at(level: float) -> Dict[str, float]:
            return {n: float(torch.quantile(v, level, interpolation="higher")) for n, v in base.items()}

        if mode == "open_loop":
            self.last_calibration = {"mode": mode, "level": 1.0 - alpha, "rate": rate, "passes": 1, "simulated": 1,
                                     "within_tolerance": True, "alpha": alpha}
            return at(1.0 - alpha)
        tolerance = float(core.get(settings, "calibration.tolerance"))      # of alpha, as mock_engine
        max_passes = int(core.get(settings, "calibration.max_passes"))
        lo, hi, level, seen = 0.0, 1.0, 1.0 - alpha, []
        for n in range(1, max_passes + 1):
            thresholds = at(level)
            _, rate = self._pass(units, arm, settings, env, store, thresholds, f".p{n}")
            seen.append((level, rate, thresholds))
            if abs(rate - alpha) <= tolerance * alpha or hi - lo < 1e-12:
                break
            lo, hi = (level, hi) if rate > alpha else (lo, level)          # a higher level means fewer denies
            level = 0.5 * (lo + hi)
        below = [x for x in seen if x[1] <= alpha * (1.0 + tolerance)]
        best = max(below, key=lambda x: x[1]) if below else min(seen, key=lambda x: x[1])
        self.last_calibration = {"mode": mode, "level": best[0], "rate": best[1], "passes": n, "simulated": n + 1,
                                 "within_tolerance": abs(best[1] - alpha) <= tolerance * alpha, "alpha": alpha}
        return best[2]

    def _train(self, unit: Mapping, settings: Mapping, env: Env, store) -> None:
        """A gnn_train unit: attacked conversations without a defence, logged for the GNN (spec 2.7). The unit is one
        (edge probability, attackers) setting; its conversations are spread over `gnn_train.graphs_per_setting` random
        graphs with their own attackers and over questions that no evaluation or calibration unit uses (D104)."""
        derived = {**unit, "topology": "random_directed", "seed": core.seed_of("gnn_train", unit["id"]) % 2 ** 31,
                   "audit": "uncapped", "intervention": "uncapped"}
        local = copy.deepcopy(dict(settings))
        core.put(local, "runtime.random_edge_probability", unit["edge_probability"])
        core.put(local, "runtime.n_insiders", unit["attackers"])
        core.put(local, "runtime.rate_limit_per_identity", 1)
        core.put(local, "runtime.llm_sink_positions.random_directed", [0])
        SeedRun(derived, NO_DEFENCE, local, env, store, calibrating=False,
                on_record=lambda record: gnn.log_conversations(settings, dict(unit), [record])).run()
