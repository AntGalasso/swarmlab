"""Fakes for the runner's tests: a platform that writes progressive rows, arm specs, a recording tracker.

The runner never builds a defender, so every ArmSpec factory here fails loudly if it is called.
"""
from __future__ import annotations

from types import MappingProxyType

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx


def _no_defender():
    raise AssertionError("the runner must never build a defender")


def _arm(name, platforms=("mock", "llm"), **flags):
    return core.ArmSpec(name, _no_defender, platforms=platforms, **flags)


# The flags are those of the real features: A, oracle_alloc, GNN (published cut) and BAZT_upstream report no calibrated
# statistic; B, C, BAZT, the allocation baselines and the GNN stand-in do.
ARMS = {a.name: a for a in [
    _arm("A"), _arm("B", calibrated=True), _arm("C", calibrated=True), _arm("BAZT", calibrated=True),
    _arm("BAZT_upstream", calibrated=True), _arm("BAZT_no_budget", calibrated=True),
    _arm("BAZT_no_topology", calibrated=True), _arm("BAZT_no_graduated", calibrated=True), _arm("GNN_standin", platforms=("mock",), calibrated=True),
    _arm("GNN", platforms=("llm",)), _arm("GNN_recalibrated", platforms=("llm",), calibrated=True),
    _arm("random_alloc", calibrated=True),
    _arm("oracle_alloc", needs_truth=True), _arm("bra_greedy", calibrated=True),
]}


def threshold_of(cell) -> float:
    """A deterministic, cell-dependent threshold, so a test can tell which cell a unit's thresholds came from."""
    return sum(ord(c) for c in "|".join(map(str, cell))) / 1000


class FakePlatform:
    """Writes rows the way a real platform must: appended as produced, resumed after the last saved row, then done."""

    def __init__(self, name="mock", rows_per_unit=1, crash_on=None, calibrate_result=None, raises=None):
        self.name = name
        self.rows_per_unit = rows_per_unit
        self.crash_on = crash_on                  # unit -> bool: crash after the first row of that unit
        self.calibrate_result = calibrate_result  # overrides the returned thresholds
        self.diagnostics = None                   # what last_calibration reports after a calibrate call
        self.last_calibration = None
        self.raises = raises                      # exception raised by run_batch before anything is written
        self.batches = []                         # (arm name, units as received)
        self.probes = []                          # preflight probes: (arm name, units, store root name)
        self.probe_raises = None                  # exception raised by a preflight probe
        self.calibrations = []
        self.trackers = []

    def run_batch(self, units, arm, settings, store, tracker):
        if store.root.name == "preflight-probe":
            self.probes.append((arm.name, [dict(u) for u in units], store.root.name))
            if self.probe_raises:
                raise self.probe_raises
            return []
        self.batches.append((arm.name, [dict(u) for u in units]))
        self.trackers.append(tracker)
        if self.raises:
            raise self.raises
        rows = []
        for u in units:
            if store.is_done(u["id"]):
                continue
            for k in range(len(store.resume_rows(u["id"])), self.rows_per_unit):
                row = {"unit": u, "k": k, "accuracy": 1.0}
                store.append_rows(u["id"], [row])
                rows.append(row)
                if self.crash_on and self.crash_on(u) and k == 0:
                    raise RuntimeError("simulated crash")
            store.mark_done(u["id"])
        return rows

    def calibrate(self, units, arm, settings, store):
        self.calibrations.append((arm.name, [dict(u) for u in units]))
        self.last_calibration = dict(self.diagnostics) if self.diagnostics is not None else None
        if self.crash_on and self.crash_on(units[0]):
            raise RuntimeError("simulated crash")
        for u in units:
            store.append_rows(u["id"], [{"unit": u, "accuracy": 1.0}])
            store.mark_done(u["id"])
        if self.calibrate_result is not None:
            return self.calibrate_result
        return {"deny_score": threshold_of(rx.cell_key(units[0]))}


class RecordingTracker:
    def __init__(self, settings=None):
        self.settings = settings                  # what the composition root built the tracker with
        self.events = []
        self.run_id = "run-1"

    def start(self, block, config, resume_id):
        self.events.append(("start", block, dict(config), resume_id))
        return resume_id or self.run_id

    def log(self, metrics, step):
        self.events.append(("log", dict(metrics), step))

    def log_rows(self, table, rows):
        self.events.append(("log_rows", table, list(rows)))

    def artifact(self, path, name, kind):
        self.events.append(("artifact", str(path), name, kind))

    def finish(self):
        self.events.append(("finish",))

    def kinds(self):
        return [e[0] for e in self.events]


class FakeAdaptive:
    """The adaptive feature's public surface: run_search returns a table that annotates units with attack_params."""

    PARAMS = MappingProxyType({"start_frac": 0.5})

    def __init__(self):
        self.searches = []                        # (units, thresholds of the first cell, out_dir)

    def search_episodes(self, units, settings):
        return 7 * len(list(units))

    def run_search(self, units, arm_of, platform, settings, out_dir, thresholds_of=None, tracker=None):
        units = list(units)
        cell = {k: units[0][k] for k in ("arm", "topology", "agents")}
        self.searches.append((units, dict(thresholds_of(cell)), out_dir))
        arm_of(units[0]["arm"])
        return self

    def annotate(self, unit):
        return {**unit, "attack_params": {} if unit["attack"] == "none" else dict(self.PARAMS)}


class FakeAnalysis:
    """The analysis feature's public surface."""

    def __init__(self, missing=()):
        self.missing, self.calls = list(missing), []

    def missing_keys(self, settings):
        return list(self.missing)

    def analyze(self, results_dir, settings, out=None):
        self.calls.append(results_dir)
        return {}


class FakeSync:
    """The sync feature's public surface: start(settings, paths) -> handle with stop()."""

    def __init__(self):
        self.started, self.stopped = [], 0

    def start(self, settings, paths):
        self.started.append((settings, list(paths)))
        outer = self

        class Handle:
            def stop(self):
                outer.stopped += 1
        return Handle()


def wiring(platforms=None, arms=None, tracker=None, adaptive=None, analysis=None, sync=None):
    """A Wiring over fakes. `tracker` is returned by Wiring.tracker only when tracking is enabled."""
    platforms = {"mock": FakePlatform("mock"), "llm": FakePlatform("llm")} if platforms is None else platforms
    arms = ARMS if arms is None else arms

    def arm(name):
        if name not in arms:
            raise rx.FeatureMissing("conventional", f"no arm named {name!r}")
        return arms[name]

    def platform(name):
        if name not in platforms:
            raise rx.FeatureMissing("mock_engine" if name == "mock" else "llm_harness", f"no platform {name!r}")
        return platforms[name]

    def adaptive_feature():
        if adaptive is None:
            raise rx.FeatureMissing("adaptive", "no module named bazt_lab.features.adaptive")
        return adaptive

    def optional(feature, module):
        def get():
            if module is None:
                raise rx.FeatureMissing(feature, f"no module named bazt_lab.features.{feature}")
            return module
        return get

    return rx.Wiring(arm=arm, platform=platform, tracker=lambda settings: tracker, adaptive=adaptive_feature,
                     analysis=optional("analysis", analysis), sync=optional("sync", sync))


# ------------------------------------------------------------------------------ settings for the tests
# Test values for the settings B, C and BAZT need before their first unit (not experiment settings).
NEEDED_MOCK = ["runtime.sanitize_success=0.9", "runtime.insider_bias=5.0", "runtime.ta_detectability=0.5",
               "runtime.ma_detectability=0.5", "runtime.aitm_rewrite_strength=0.5", "calibration.mode='closed_loop'", "conventional.c.judge_score_bins.mock=[0.0, 1.0]", "conventional.c.presence_window_rounds=7",
               "mock.auditor.binormal_method='shared_threshold'"]
NEEDED_LLM = ["calibration.mode='closed_loop'", "conventional.c.judge_score_bins.llm=[-3.0, -1.0]", "conventional.c.presence_window_rounds=7",
              "llm.judge.provider='p'", "llm.judge.top_logprobs=20", "llm.judge.floor_probability=1e-6"]
# Test values for the run-time variables, not experiment settings.
PRE_REGISTERED = ["runtime.ni_margin_accuracy=0.01", "runtime.rank_agreement_threshold=0.8"]
RUNTIME_MOCK = ["runtime.n_insiders=3", "runtime.rate_limit_per_identity=1", "runtime.horizon_constant=3",
                "runtime.eps_correct=1.0", "runtime.recovery_margin=0.01", "runtime.ba_m=2", "runtime.ws_k=4",
                "runtime.ws_rewire_p=0.25"] + PRE_REGISTERED + NEEDED_MOCK
RUNTIME_LLM = ["runtime.n_insiders=3", "runtime.rate_limit_per_identity=1", "runtime.recovery_margin=0.01",
               "runtime.random_edge_probability=0.3", "runtime.ba_m=2", "runtime.ws_k=4", "runtime.ws_rewire_p=0.25",
               "runtime.llm_seeds=[1, 2]", "runtime.llm_calibration_seeds=[101, 102]",
               "runtime.llm_sink_positions.hub_star=[0]"] + PRE_REGISTERED + NEEDED_LLM

# A mock frontier of 12 units per arm cell: 1 topology x (infection, none) x 2 seeds x 3 sinks.
TINY_MOCK = ["mock.topologies=['chain']", "mock.families=['infection']", "mock.main_seeds=[1, 2]",
             "mock.heldout_seeds=[31, 32]", "mock.tune_seeds=[9001, 9002]", "mock.frontier_agents=[3]",
             "mock.epidemiology_agents=[3]", "mock.batch_size=4"]
ONLY_A = ["mock.uncapped_arms=['A']", "mock.capped_arms=[]", "mock.allocation_arms=[]"]
CALIBRATED_B = ["mock.calibrated_arms=['B']", "mock.uncapped_arms=['A', 'B']", "mock.capped_arms=['B']",
                "mock.allocation_arms=[]", "budgets.levels_percent=[10]"]
# One LLM unit per (arm, backbone, topology, attack cell, seed).
TINY_LLM = ["llm.backbones=['bb1', 'bb2']", "llm.topologies=['hub_star']", "llm.attack_cells=[['insider_pi', 'mmlu']]",
            "llm.attack_free_tasks=['mmlu']", "llm.seeds_per_cell=2", "llm.conversations_per_cell=10",
            "llm.calibration_conversations_per_cell=10", "llm.headline_arms=['A', 'B']", "llm.capped_arms=['B']",
            "llm.calibrated_arms=['B']", "llm.capped_cells=[[10, 10]]"]
# Every scoped mock block set, so every block of the registry can be expanded.
SCOPED = ["mock.epidemiology.topologies=['chain']", "mock.epidemiology.arms=['A', 'BAZT']",
          "mock.epidemiology.cells=[[10, 10]]", "mock.ablations.arms=['BAZT_no_budget']",
          "mock.ablations.cells=[[10, 10]]", "mock.adaptive.arms=['BAZT']", "mock.adaptive.cells=[[10, 10]]",
          "mock.presets_robustness.arms=['C']", "mock.presets_robustness.cells=[[10, 10]]",
          "mock.costs_robustness.arms=['C', 'BAZT']", "mock.costs_robustness.cells=[[10, 10], [2, 2]]",
          "mock.legacy.arms=['fixed_early']", "mock.legacy.cells=[[10, 10]]",
          "llm.adaptive_check.arms=['B']", "llm.adaptive_check.families=['insider_pi']",
          "llm.adaptive_check.condition='gray_box'", "gnn_train.backbone='bb1'"]


def settings(*groups, profile=None):
    """Settings from profiles.toml with each group of `key=value` overrides applied in order."""
    return rx.load_settings(profile=profile, overrides=[o for g in groups for o in g])
