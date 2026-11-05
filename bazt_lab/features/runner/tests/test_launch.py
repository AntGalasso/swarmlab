"""Launching blocks over fakes: batching, resume, calibration and threshold propagation, tracker, refusals."""
import hashlib
import json
import re

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.tests import fakes

MOCK_A = [fakes.RUNTIME_MOCK, fakes.TINY_MOCK, fakes.ONLY_A]
MOCK_B = [fakes.RUNTIME_MOCK, fakes.TINY_MOCK, fakes.CALIBRATED_B]
LLM = [fakes.RUNTIME_LLM, fakes.TINY_LLM]


def launch(settings, block, tmp_path, platform=None, platform_name="mock", allow_live_llm=False, **wiring_kw):
    plat = platform or fakes.FakePlatform(platform_name)
    rx.run_block(settings, block, allow_live_llm, wiring=fakes.wiring({platform_name: plat}, **wiring_kw),
                 results_root=tmp_path)
    return plat


def block_dir(tmp_path, block, profile="sprint"):
    return tmp_path / profile / block


def past_kickoff(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2020-01-01T00:00:00+00:00")


# ===================================================================================== batching and resume
def test_a_block_runs_in_batches_of_one_group_chunked_at_batch_size(tmp_path):
    plat = launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path)
    assert [len(units) for _, units in plat.batches] == [4, 2, 4, 2]
    assert {arm for arm, _ in plat.batches} == {"A"}
    for _, units in plat.batches:
        assert len({rx.group_key(u) for u in units}) == 1
    sent = [u for _, units in plat.batches for u in units]
    assert len({u["id"] for u in sent}) == 12 and all(u["id"] == rx.unit_id(u) for u in sent)
    assert all(u["thresholds"] == {} for u in sent)
    assert len(list((block_dir(tmp_path, "mock_frontier") / "units").glob("*.done"))) == 12


def test_the_batch_size_is_a_setting_and_a_missing_one_is_refused(tmp_path):
    s = fakes.settings(*MOCK_A, ["mock.batch_size=100"])
    assert [len(u) for _, u in launch(s, "mock_frontier", tmp_path).batches] == [6, 6]
    bare = fakes.settings(*MOCK_A)
    del bare["mock"]["batch_size"]
    with pytest.raises(SystemExit, match=r"mock.batch_size"):
        launch(bare, "mock_frontier", tmp_path / "other")


def test_llm_blocks_run_one_unit_per_batch(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(*LLM, ["llm.headline_arms=['A']"])
    plat = launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    assert len(plat.batches) == 8 and {len(u) for _, u in plat.batches} == {1}


def test_a_platform_that_returns_without_marking_its_units_done_is_caught_at_once(tmp_path):
    """Unmarked units would be run again, and their rows appended twice, at the next launch."""
    tracker = fakes.RecordingTracker()

    class Forgetful(fakes.FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            store.append_rows(units[0]["id"], [{"k": 0}])
            store.mark_done(units[0]["id"])
            return []                                     # the other units of the batch are neither run nor marked

    s = fakes.settings(*MOCK_A, ["tracking.enabled=true"])
    with pytest.raises(RuntimeError, match=r"mock platform.*without marking 3 of 4 units done.*A-"):
        launch(s, "mock_frontier", tmp_path, platform=Forgetful("mock"), tracker=tracker)
    assert not [e for e in tracker.events if e[0] == "log"]          # no progress is reported for the bad batch
    assert tracker.kinds()[-1] == "finish"


def test_a_finished_block_is_skipped_on_the_next_launch(tmp_path, capsys):
    s = fakes.settings(*MOCK_A)
    launch(s, "mock_frontier", tmp_path)
    again = launch(s, "mock_frontier", tmp_path)
    assert again.batches == []
    assert "0 units run, 12 already done" in capsys.readouterr().out


def test_resume_after_a_crash_continues_after_the_last_saved_row(tmp_path):
    s = fakes.settings(*MOCK_A)
    crashing = fakes.FakePlatform("mock", rows_per_unit=2, crash_on=lambda u: u["seed"] == 2 and u["sink"] == 1)
    with pytest.raises(RuntimeError, match="simulated crash"):
        launch(s, "mock_frontier", tmp_path, platform=crashing)
    store = core.CheckpointStore(block_dir(tmp_path, "mock_frontier"))
    crashed = next(u["id"] for _, units in crashing.batches for u in units if not store.is_done(u["id"])
                   and store.resume_rows(u["id"]))
    assert len(store.resume_rows(crashed)) == 1
    finished = {u["id"] for _, units in crashing.batches for u in units if store.is_done(u["id"])}
    assert finished and crashed not in finished

    healthy = launch(s, "mock_frontier", tmp_path, platform=fakes.FakePlatform("mock", rows_per_unit=2))
    resent = {u["id"] for _, units in healthy.batches for u in units}
    assert crashed in resent and not (finished & resent)         # done units are never handed over again
    ids = {rx.unit_id(u) for u in rx.units_mock_frontier(s, False)}
    store = core.CheckpointStore(block_dir(tmp_path, "mock_frontier"))      # a store reads the marks it finds at creation
    assert len(ids) == 12 and all(store.is_done(i) and len(store.resume_rows(i)) == 2 for i in ids)


# ============================================================================== calibration and thresholds
def calibrate_then(tmp_path, s, platform=None):
    cal = launch(s, "mock_calibration", tmp_path, platform=platform)
    return cal


def test_calibration_calls_the_platform_once_per_cell_and_writes_thresholds_json(tmp_path):
    cal = calibrate_then(tmp_path, fakes.settings(*MOCK_B))
    assert [(arm, len(units)) for arm, units in cal.calibrations] == [("B", 2)]
    assert {(u["audit"], u["intervention"], u["attack"]) for u in cal.calibrations[0][1]} == {
        ("uncapped", "uncapped", "none")}
    saved = json.loads((block_dir(tmp_path, "mock_calibration") / "thresholds.json").read_text())
    assert saved["alpha"] == 0.02
    assert saved["cells"] == [{"arm": "B", "topology": "chain", "agents": 3, "backbone": None, "task": None,
                               "thresholds": {"deny_score": fakes.threshold_of(("B", "chain", 3, None, None))}}]


def test_evaluation_units_carry_the_thresholds_of_their_cell_at_every_budget(tmp_path):
    s = fakes.settings(*MOCK_B)
    calibrate_then(tmp_path, s)
    plat = launch(s, "mock_frontier", tmp_path)
    sent = [u for _, units in plat.batches for u in units]
    expected = {"deny_score": fakes.threshold_of(("B", "chain", 3, None, None))}
    b_units = [u for u in sent if u["arm"] == "B"]
    assert len({(u["audit"], u["intervention"]) for u in b_units}) == 4          # uncapped + 3 capped cells
    assert all(u["thresholds"] == expected for u in b_units)
    assert all(u["thresholds"] == {} for u in sent if u["arm"] == "A")           # A has no tunable score


def test_a_calibrated_arm_without_thresholds_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(SystemExit, match=r"no calibrated thresholds.*arm=B.*mock_calibration"):
        launch(fakes.settings(*MOCK_B), "mock_frontier", tmp_path)
    assert not any(tmp_path.iterdir())


def test_a_calibrated_arm_the_calibration_block_does_not_calibrate_is_refused_with_the_setting_to_change(tmp_path):
    """random_alloc reports a decision statistic (ArmSpec.calibrated), so without a threshold it would collect the
    statistic and never act. Running the calibration block again cannot help: the block does not list the arm."""
    s = fakes.settings(*MOCK_B, ["mock.uncapped_arms=['A', 'B', 'random_alloc']"])
    calibrate_then(tmp_path, s)
    with pytest.raises(SystemExit, match=r"random_alloc.*mock\.calibrated_arms"):
        launch(s, "mock_frontier", tmp_path)
    assert not block_dir(tmp_path, "mock_frontier").exists()


def test_the_same_refusal_names_the_llm_setting(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(*LLM, ["llm.headline_arms=['A', 'random_alloc']"])
    with pytest.raises(SystemExit, match=r"random_alloc.*llm\.calibrated_arms"):
        launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    assert not any(tmp_path.iterdir())


def test_a_cell_that_was_not_calibrated_is_refused_and_named(tmp_path):
    calibrate_then(tmp_path, fakes.settings(*MOCK_B))
    wider = fakes.settings(*MOCK_B, ["mock.frontier_agents=[3, 4]"])
    with pytest.raises(SystemExit, match=r"agents=4"):
        launch(wider, "mock_frontier", tmp_path)
    assert not block_dir(tmp_path, "mock_frontier").exists()


def test_the_uncalibrated_arms_need_no_thresholds(tmp_path):
    plat = launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path)
    assert plat.batches


def test_an_interrupted_calibration_resumes_with_the_cells_left(tmp_path):
    s = fakes.settings(*MOCK_B, ["mock.topologies=['chain', 'star_hub']"])
    crashing = fakes.FakePlatform("mock", crash_on=lambda u: u["topology"] == "star_hub")
    with pytest.raises(RuntimeError, match="simulated crash"):
        calibrate_then(tmp_path, s, crashing)
    path = block_dir(tmp_path, "mock_calibration") / "thresholds.json"
    assert [c["topology"] for c in json.loads(path.read_text())["cells"]] == ["chain"]
    healthy = calibrate_then(tmp_path, s)
    assert [units[0]["topology"] for _, units in healthy.calibrations] == ["star_hub"]
    assert [c["topology"] for c in json.loads(path.read_text())["cells"]] == ["chain", "star_hub"]


def test_llm_thresholds_are_per_backbone_and_task(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(*LLM, ["llm.attack_free_tasks=['mmlu', 'gsm8k']", "llm.attack_cells=[['insider_pi', 'mmlu']]",
                              "llm.headline_arms=['B']"])
    cal = launch(s, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    assert sorted(rx.cell_key(units[0]) for _, units in cal.calibrations) == [
        ("B", "hub_star", 12, "bb1", "gsm8k"), ("B", "hub_star", 12, "bb1", "mmlu"),
        ("B", "hub_star", 12, "bb2", "gsm8k"), ("B", "hub_star", 12, "bb2", "mmlu")]
    assert all(len(units) == 2 for _, units in cal.calibrations)                  # the 2 calibration seeds
    ev = launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    for _, (unit,) in ev.batches:
        assert unit["thresholds"] == {"deny_score": fakes.threshold_of(rx.cell_key(unit))}
    assert any(u["task"] == "mmlu" and u["backbone"] == "bb2" for _, (u,) in ev.batches)


def test_llm_evaluation_without_its_calibration_is_refused(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    with pytest.raises(SystemExit, match=r"llm_calibration"):
        launch(fakes.settings(*LLM), "llm_capped", tmp_path, platform_name="llm", allow_live_llm=True)


def test_calibrating_an_arm_that_reports_no_statistic_is_refused(tmp_path):
    with pytest.raises(SystemExit, match=r"'A'.*not calibrated"):
        launch(fakes.settings(*MOCK_B, ["mock.calibrated_arms=['A']"]), "mock_calibration", tmp_path)


def test_a_calibration_that_returns_nothing_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="returned no thresholds"):
        calibrate_then(tmp_path, fakes.settings(*MOCK_B), fakes.FakePlatform("mock", calibrate_result={}))
    assert not (block_dir(tmp_path, "mock_calibration") / "thresholds.json").exists()


def test_gnn_training_goes_through_the_llm_platform_without_thresholds(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(fakes.RUNTIME_LLM, fakes.TINY_LLM, ["gnn_train.backbone='bb1'"])
    plat = launch(s, "gnn_train", tmp_path, platform_name="llm", allow_live_llm=True)
    assert len(plat.batches) == 100 and {arm for arm, _ in plat.batches} == {"GNN"}
    assert all(u["thresholds"] == {} for _, (u,) in plat.batches)


def test_a_calibration_with_a_threshold_that_is_not_finite_is_refused(tmp_path):
    for bad in (float("nan"), float("inf"), float("-inf")):
        platform = fakes.FakePlatform("mock", calibrate_result={"deny_score": bad})
        with pytest.raises(SystemExit, match=r"not finite.*deny_score"):
            calibrate_then(tmp_path, fakes.settings(*MOCK_B), platform)
        assert not (block_dir(tmp_path, "mock_calibration") / "thresholds.json").exists()


def test_a_calibration_is_not_resumed_at_another_alpha(tmp_path):
    chain_star = ["mock.topologies=['chain', 'star_hub']"]
    crashing = fakes.FakePlatform("mock", crash_on=lambda u: u["topology"] == "star_hub")
    with pytest.raises(RuntimeError, match="simulated crash"):
        calibrate_then(tmp_path, fakes.settings(*MOCK_B, chain_star), crashing)
    path = block_dir(tmp_path, "mock_calibration") / "thresholds.json"
    before = path.read_text()
    with pytest.raises(SystemExit, match=r"alpha=0.02.*alpha=0.05"):
        calibrate_then(tmp_path, fakes.settings(*MOCK_B, chain_star, ["budgets.alpha=0.05"]))
    assert path.read_text() == before


# ===================================================================================================== refusals
def test_a_missing_setting_of_the_block_is_refused_with_its_key(tmp_path):
    with pytest.raises(SystemExit, match=r"Refusing mock_ablations.*mock.ablations.arms"):
        launch(fakes.settings(fakes.RUNTIME_MOCK, fakes.TINY_MOCK), "mock_ablations", tmp_path)


def test_a_setting_the_platform_needs_is_refused_with_its_key_and_the_tracker_is_closed(tmp_path):
    tracker = fakes.RecordingTracker()
    s = fakes.settings(*MOCK_A, ["tracking.enabled=true"])
    boom = fakes.FakePlatform("mock", raises=core.Unset("runtime.insider_bias"))
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier.*runtime.insider_bias"):
        launch(s, "mock_frontier", tmp_path, platform=boom, tracker=tracker)
    assert tracker.kinds()[-1] == "finish"


def test_a_feature_that_is_not_built_is_named_with_the_block_that_needs_it(tmp_path):
    arms = {k: v for k, v in fakes.ARMS.items() if k != "A"}
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: feature conventional is not built"):
        launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path, arms=arms)
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: feature mock_engine is not built"):
        rx.run_block(fakes.settings(*MOCK_A), "mock_frontier", False, wiring=fakes.wiring({}), results_root=tmp_path)
    assert not any(tmp_path.iterdir())


def test_tracking_without_the_tracking_feature_and_an_arm_no_feature_exports_are_refused_by_name(tmp_path):
    base = fakes.wiring({"mock": fakes.FakePlatform("mock")})

    def no_tracker(settings):
        raise rx.FeatureMissing("tracking", "No module named 'bazt_lab.features.tracking'")

    def no_such_arm(name):
        raise LookupError(f"no feature exports an arm named {name!r}")

    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: feature tracking is not built"):
        rx.run_block(fakes.settings(*MOCK_A, ["tracking.enabled=true"]), "mock_frontier", False,
                     wiring=rx.Wiring(base.arm, base.platform, no_tracker), results_root=tmp_path)
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: no feature exports an arm named 'A'"):
        rx.run_block(fakes.settings(*MOCK_A), "mock_frontier", False,
                     wiring=rx.Wiring(no_such_arm, base.platform, base.tracker), results_root=tmp_path)
    assert not any(tmp_path.iterdir())


def test_an_arm_is_refused_on_a_platform_it_does_not_run_on(tmp_path):
    s = fakes.settings(*MOCK_A, ["mock.uncapped_arms=['GNN']"])
    with pytest.raises(SystemExit, match=r"arm 'GNN' does not run on the mock platform"):
        launch(s, "mock_frontier", tmp_path)


def test_llm_blocks_refuse_without_the_live_llm_flag(tmp_path):
    with pytest.raises(SystemExit, match="allow-live-llm"):
        launch(fakes.settings(*LLM), "llm_capped", tmp_path, platform_name="llm")
    assert not any(tmp_path.iterdir())


def test_llm_blocks_refuse_before_kickoff(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2999-01-01T00:00:00+00:00")
    with pytest.raises(SystemExit, match="before kickoff"):
        launch(fakes.settings(*LLM), "llm_capped", tmp_path, platform_name="llm", allow_live_llm=True)
    assert not any(tmp_path.iterdir())


def test_the_freeze_marker_lives_in_the_results_root(tmp_path):
    s = fakes.settings(*MOCK_A)
    with pytest.raises(SystemExit, match="not frozen"):
        launch(s, "mock_heldout", tmp_path)
    (tmp_path / "ANALYSIS_FROZEN").write_text("frozen by hand")
    plat = launch(s, "mock_heldout", tmp_path)
    assert {u["seed"] for _, units in plat.batches for u in units} == {31, 32}


def test_a_block_that_is_already_running_is_refused_and_a_finished_one_is_free_again(tmp_path):
    s = fakes.settings(*MOCK_A)
    refused = []

    class Reentrant(fakes.FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            if not refused and store.root.name != "preflight-probe":
                refused.append(True)
                with pytest.raises(SystemExit, match=r"another launch.*mock_frontier"):
                    launch(s, "mock_frontier", tmp_path)          # second launch, same results root and block
            return super().run_batch(units, arm, settings, store, tracker)

    launch(s, "mock_frontier", tmp_path, platform=Reentrant("mock"))
    assert refused and len(list((block_dir(tmp_path, "mock_frontier") / "units").glob("*.done"))) == 12
    assert launch(s, "mock_frontier", tmp_path).batches == []     # the lock was released: nothing left to run


# ===================================================================================================== manifest
def test_the_manifest_records_the_run_before_any_result_exists(tmp_path):
    s = fakes.settings(*MOCK_B)
    calibrate_then(tmp_path, s)
    launch(s, "mock_frontier", tmp_path)
    (path,) = block_dir(tmp_path, "mock_frontier").glob("manifest-*.json")
    m = json.loads(path.read_text())
    assert (m["block"], m["profile"], m["settings"]["profile"]) == ("mock_frontier", "sprint", "sprint")
    assert m["pre_registered"] == {"ni_margin_accuracy": 0.01, "rank_agreement_threshold": 0.8}
    assert m["versions"]["torch"] == torch.__version__ and set(m["versions"]) == {"python", "torch", "numpy"}
    assert m["threads"] == torch.get_num_threads()
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", m["git_commit"])
    assert m["started_at"].endswith("+00:00") and m["units"] == len(rx.units_mock_frontier(s, False))
    thresholds = block_dir(tmp_path, "mock_calibration") / "thresholds.json"
    assert m["thresholds_sha256"] == hashlib.sha256(thresholds.read_bytes()).hexdigest()


def test_a_manifest_without_thresholds_says_so(tmp_path):
    launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path)
    (path,) = block_dir(tmp_path, "mock_frontier").glob("manifest-*.json")
    assert json.loads(path.read_text())["thresholds_sha256"] is None


# ====================================================================================================== tracker
def test_tracking_is_off_unless_enabled(tmp_path):
    tracker = fakes.RecordingTracker()
    plat = launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path, tracker=tracker)
    assert tracker.events == [] and {type(t) for t in plat.trackers} == {core.NullTracker}


def test_tracking_enabled_must_be_a_boolean(tmp_path):
    with pytest.raises(SystemExit, match=r"tracking.enabled"):
        launch(fakes.settings(*MOCK_A, ["tracking.enabled='yes'"]), "mock_frontier", tmp_path)


def test_one_tracking_run_per_block_logs_progress_per_batch_and_the_manifest(tmp_path):
    tracker = fakes.RecordingTracker()
    s = fakes.settings(*MOCK_A, ["tracking.enabled=true"])
    plat = launch(s, "mock_frontier", tmp_path, tracker=tracker)
    kinds = tracker.kinds()
    assert kinds[0] == "start" and kinds[-1] == "finish" and kinds.count("start") == 1
    _, block, config, resume_id = tracker.events[0]
    assert block == "mock_frontier" and resume_id is None
    assert core.get(config, "profile") == "sprint" and config["manifest"]["block"] == "mock_frontier"
    logs = [e for e in tracker.events if e[0] == "log"]
    assert [e[2] for e in logs] == [4, 6, 10, 12]                                 # units completed so far
    assert logs[-1][1]["runner/units_done"] == 12 and logs[-1][1]["runner/batches"] == 4
    (artifact,) = [e for e in tracker.events if e[0] == "artifact"]
    assert artifact[1].endswith(".json") and "manifest-" in artifact[1] and artifact[3] == "manifest"
    assert (block_dir(tmp_path, "mock_frontier") / "tracker_run_id.txt").read_text() == "run-1"
    assert set(plat.trackers) == {tracker}


class TrackingError(RuntimeError):
    """Stands in for the tracking feature's error: a RuntimeError whose message names the setting or variable."""


def test_a_tracker_that_cannot_be_set_up_refuses_the_launch_before_anything_is_written(tmp_path):
    base = fakes.wiring({"mock": fakes.FakePlatform("mock")})

    def refusing(settings):
        raise TrackingError("tracking.mode must be online, offline or disabled, got 'sideways'")

    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: tracking\.mode must be online"):
        rx.run_block(fakes.settings(*MOCK_A, ["tracking.enabled=true"]), "mock_frontier", False,
                     wiring=rx.Wiring(base.arm, base.platform, refusing), results_root=tmp_path)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("failure, message", [
    (TrackingError("WANDB_API_KEY is not set"), r"Refusing mock_frontier: WANDB_API_KEY is not set"),
    (core.Unset("tracking.project"), r"Refusing mock_frontier: .*tracking\.project"),
])
def test_a_tracking_run_that_cannot_start_stops_the_launch_and_leaves_no_manifest(tmp_path, failure, message):
    class CannotStart(fakes.RecordingTracker):
        def start(self, block, config, resume_id):
            raise failure

    s = fakes.settings(*MOCK_A, ["tracking.enabled=true"])
    plat = fakes.FakePlatform("mock")
    with pytest.raises(SystemExit, match=message):
        launch(s, "mock_frontier", tmp_path, platform=plat, tracker=CannotStart())
    assert plat.batches == []
    out = block_dir(tmp_path, "mock_frontier")
    assert not list(out.glob("manifest-*.json")) and not (out / "tracker_run_id.txt").exists()
    assert launch(s, "mock_frontier", tmp_path, tracker=fakes.RecordingTracker()).batches    # the lock was released


def test_a_resumed_block_continues_its_tracking_run_and_a_crash_still_closes_it(tmp_path):
    tracker = fakes.RecordingTracker()
    s = fakes.settings(*MOCK_A, ["tracking.enabled=true"])
    crashing = fakes.FakePlatform("mock", crash_on=lambda u: u["seed"] == 2)
    with pytest.raises(RuntimeError):
        launch(s, "mock_frontier", tmp_path, platform=crashing, tracker=tracker)
    assert tracker.kinds()[-1] == "finish"
    again = fakes.RecordingTracker()
    launch(s, "mock_frontier", tmp_path, tracker=again)
    assert again.events[0][3] == "run-1"
    steps = [e[2] for e in again.events if e[0] == "log"]
    assert steps == sorted(steps) and steps[-1] == 12


# ============================================================================================== fail-fast preflight
def without(settings, dotted):
    *path, leaf = dotted.split(".")
    node = settings
    for part in path:
        node = node[part]
    del node[leaf]
    return settings


@pytest.mark.parametrize("arm, key", [
    ("B", "mock.auditor.binormal_method"), ("C", "mock.auditor.binormal_method"),
    ("C", "conventional.c.presence_window_rounds"), ("C", "conventional.c.judge_score_bins.mock"),
    ("BAZT", "conventional.c.judge_score_bins.mock"), ("BAZT_no_budget", "conventional.c.presence_window_rounds"),
])
def test_a_mock_block_resolves_the_keys_of_its_arms_before_its_first_unit(tmp_path, arm, key):
    s = fakes.settings(*MOCK_A, [f"mock.uncapped_arms=['{arm}']", f"mock.calibrated_arms=['{arm}']"])
    arms = {**fakes.ARMS, arm: fakes.ARMS.get(arm) or fakes._arm(arm, calibrated=True)}
    plat = fakes.FakePlatform("mock")
    with pytest.raises(SystemExit, match=rf"Refusing mock_calibration: .*{key}"):
        launch(without(s, key), "mock_calibration", tmp_path, platform=plat, arms=arms)
    assert plat.calibrations == [] and not any(tmp_path.iterdir())


def test_an_arm_without_judge_or_trust_attributes_needs_none_of_those_keys(tmp_path):
    s = fakes.settings(*MOCK_A)
    for key in ("mock.auditor.binormal_method", "conventional.c.presence_window_rounds",
                "conventional.c.judge_score_bins.mock"):
        without(s, key)
    assert launch(s, "mock_frontier", tmp_path).batches


@pytest.mark.parametrize("key", ["llm.judge.provider", "llm.judge.top_logprobs", "llm.judge.floor_probability",
                                 "conventional.c.judge_score_bins.llm"])
def test_an_llm_block_with_judge_arms_resolves_the_judge_keys_first(tmp_path, monkeypatch, key):
    past_kickoff(monkeypatch)
    s = without(fakes.settings(*LLM, ["llm.headline_arms=['B', 'C']"]), key)
    plat = fakes.FakePlatform("llm")
    with pytest.raises(SystemExit, match=rf"Refusing llm_headline: .*{key}"):
        launch(s, "llm_headline", tmp_path, platform=plat, platform_name="llm", allow_live_llm=True)
    assert plat.batches == [] and not any(tmp_path.iterdir())


def test_an_llm_block_without_judge_arms_needs_no_judge_keys(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = without(fakes.settings(*LLM, ["llm.headline_arms=['A']"]), "llm.judge.provider")
    assert launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True).batches


def test_run_time_variables_of_every_run_are_refused_by_name(tmp_path):
    for key in ("rate_limit_per_identity", "recovery_margin"):
        s = without(fakes.settings(*MOCK_A), f"runtime.{key}")
        with pytest.raises(SystemExit, match=rf"run-time variables not set.*{key}"):
            launch(s, "mock_frontier", tmp_path)


# ============================================================================================ matched-timing arms
def test_a_matched_timing_arm_is_refused_because_no_schedule_is_recorded(tmp_path):
    arms = {**fakes.ARMS, "fixed_early_matched": fakes._arm("fixed_early_matched", platforms=("mock",))}
    s = fakes.settings(*MOCK_A, ["mock.uncapped_arms=['A', 'fixed_early_matched']"])
    plat = fakes.FakePlatform("mock")
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: arm 'fixed_early_matched'.*timing_schedule"):
        launch(s, "mock_frontier", tmp_path, platform=plat, arms=arms)
    assert plat.batches == [] and not any(tmp_path.iterdir())


# ======================================================================================================= adaptive
CAL_B = [fakes.RUNTIME_MOCK, fakes.TINY_MOCK, ["mock.calibrated_arms=['B']", "budgets.levels_percent=[10]"],
         ["mock.adaptive.arms=['B']", "mock.adaptive.cells=[[10, 10]]", "mock.adaptive.conditions=['gray_box']"]]


def test_the_adaptive_block_searches_first_and_hands_the_platform_annotated_units(tmp_path):
    s = fakes.settings(*CAL_B)
    calibrate_then(tmp_path, s)
    adaptive = fakes.FakeAdaptive()
    plat = fakes.FakePlatform("mock")
    rx.run_block(s, "mock_adaptive", False, wiring=fakes.wiring({"mock": plat}, adaptive=adaptive),
                 results_root=tmp_path)
    ((units, thresholds, out_dir),) = adaptive.searches
    assert len(units) == len(rx.BLOCKS["mock_adaptive"].units(s, True)) and "attack_params" not in units[0]
    assert thresholds == {"deny_score": fakes.threshold_of(("B", "chain", 3, None, None))}
    assert out_dir == block_dir(tmp_path, "mock_adaptive")
    sent = [u for _, us in plat.batches for u in us]
    assert {json.dumps(u["attack_params"]) for u in sent if u["attack"] != "none"} == {'{"start_frac": 0.5}'}
    assert {json.dumps(u["attack_params"]) for u in sent if u["attack"] == "none"} == {"{}"}
    assert all(u["id"] == rx.unit_id({k: v for k, v in u.items() if k != "attack_params"}) for u in sent)


def test_the_adaptive_block_without_the_adaptive_feature_is_refused_by_name(tmp_path):
    s = fakes.settings(*CAL_B)
    calibrate_then(tmp_path, s)
    with pytest.raises(SystemExit, match=r"Refusing mock_adaptive: feature adaptive is not built"):
        launch(s, "mock_adaptive", tmp_path)


def test_the_adaptive_search_that_needs_a_setting_refuses_before_anything_is_written(tmp_path):
    class NeedsKey(fakes.FakeAdaptive):
        def search_episodes(self, units, settings):
            raise core.Unset("mock.adaptive.search_seeds")

    s = fakes.settings(*CAL_B)
    calibrate_then(tmp_path, s)
    with pytest.raises(SystemExit, match=r"Refusing mock_adaptive:.*mock.adaptive.search_seeds"):
        rx.run_block(s, "mock_adaptive", False, wiring=fakes.wiring({"mock": fakes.FakePlatform("mock")},
                                                                    adaptive=NeedsKey()), results_root=tmp_path)
    assert not block_dir(tmp_path, "mock_adaptive").exists()


def test_dry_run_sizes_the_adaptive_search(capsys):
    argv = ["--dry-run", "--block", "mock_adaptive"] + [x for g in CAL_B for o in g for x in ("--set", o)]
    rx.main(argv, wiring=fakes.wiring(adaptive=fakes.FakeAdaptive()))
    n = len(rx.BLOCKS["mock_adaptive"].units(fakes.settings(*CAL_B), False))
    assert f"adaptive search: {7 * n:,} episodes" in capsys.readouterr().out


def test_the_block_names_are_those_the_analysis_reads():
    assert {"mock_frontier", "mock_epidemiology", "mock_adaptive", "mock_ablations", "llm_headline", "llm_capped",
            "llm_adaptive_check"} <= set(rx.BLOCKS)


# ====================================================================================================== llm.allow_live
def test_the_live_flag_becomes_the_llm_allow_live_setting_the_platform_reads(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    seen = []

    class Spy(fakes.FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            seen.append(core.get(settings, "llm.allow_live"))
            return super().run_batch(units, arm, settings, store, tracker)

    s = fakes.settings(*LLM, ["llm.headline_arms=['A']"])
    launch(s, "llm_headline", tmp_path, platform=Spy("llm"), platform_name="llm", allow_live_llm=True)
    assert seen and set(seen) == {True}
    assert not core.is_set(s, "llm.allow_live")                         # the caller's settings are not mutated
    (path,) = block_dir(tmp_path, "llm_headline").glob("manifest-*.json")
    assert json.loads(path.read_text())["settings"]["llm"]["allow_live"] is True


def test_a_mock_block_runs_with_allow_live_false(tmp_path):
    seen = []

    class Spy(fakes.FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            seen.append(core.get(settings, "llm.allow_live"))
            return super().run_batch(units, arm, settings, store, tracker)

    launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path, platform=Spy("mock"))
    assert seen and set(seen) == {False}


# ===================================================================================================== sizing
def test_dry_run_counts_rewrite_and_memory_attack_judge_calls(capsys):
    base = [fakes.TINY_LLM, fakes.RUNTIME_LLM, ["llm.headline_arms=['BAZT']", "llm.attack_cells=[['insider_ma', 'poisonedrag_msmarco']]",
                                               "llm.attack_free_tasks=[]",
                                               "estimates.usd_per_million_tokens.bb1=[0.1, 0.2]", "estimates.usd_per_million_tokens.bb2=[0.1, 0.2]"]]
    def calls(*extra):
        rx.main(["--dry-run", "--block", "llm_headline"] + [x for g in [*base, *extra] for o in g for x in ("--set", o)],
                wiring=fakes.wiring())
        line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("llm_headline"))
        return int(line.split()[3].replace(",", ""))
    plain = calls(["llm.sanitize_modes=['redaction']", "llm.ma_correctness=['rule']"])
    both = calls(["llm.sanitize_modes=['redaction', 'llm_rewrite']", "llm.ma_correctness=['rule', 'llm_judge']"])
    assert both > 2 * plain                       # a second sanitize unit, plus rewrite and judge calls


# =================================================================================================== gnn preflight
GNN = [fakes.RUNTIME_LLM, fakes.TINY_LLM, ["llm.headline_arms=['A', 'GNN']", "gnn.cut=0.5",
                                            "gnn.carry_flags=true", "gnn.fallback_family='pi_same_task'"]]


class FakeGnn:
    def __init__(self, finished=True):
        self.finished, self.asked = finished, []

    def deploy_family(self, attack, task, settings):
        """The gnn feature's rule: the cell's own family, else the fallback the setting names."""
        if (attack, task) in {tuple(f) for f in core.get(settings, "gnn_train.families")}:
            return f"{attack}__{task}"
        if core.get(settings, "gnn.fallback_family") == "pi_same_task":
            return f"insider_pi__{task}"
        return None

    def training_finished(self, settings, family):
        self.asked.append(family)
        return self.finished


def gnn_launch(s, tmp_path, gnn, monkeypatch):
    past_kickoff(monkeypatch)
    plat = fakes.FakePlatform("llm")
    wiring = fakes.wiring({"llm": plat})
    rx.run_block(s, "llm_headline", True, wiring=rx.Wiring(wiring.arm, wiring.platform, wiring.tracker, gnn=lambda: gnn),
                 results_root=tmp_path)
    return plat


def test_each_cell_deploys_the_gnn_of_its_own_family_and_the_fallback_covers_the_rest(tmp_path, monkeypatch):
    gnn = FakeGnn()
    assert gnn_launch(fakes.settings(*GNN), tmp_path, gnn, monkeypatch).batches
    assert set(gnn.asked) == {"insider_pi__mmlu"}              # insider_pi/mmlu itself, attack-free via the fallback


def test_a_family_without_a_gnn_and_no_fallback_is_refused_naming_the_setting(tmp_path, monkeypatch):
    s = fakes.settings(*GNN, ["gnn.fallback_family='none'"])
    with pytest.raises(SystemExit, match=r"Refusing llm_headline: no GNN family.*attack 'none'"):
        gnn_launch(s, tmp_path, FakeGnn(), monkeypatch)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("key", ["gnn.cut", "gnn.carry_flags", "gnn.fallback_family"])
def test_a_gnn_key_that_is_unset_refuses_before_any_arm_runs(tmp_path, monkeypatch, key):
    s = without(fakes.settings(*GNN), key)
    with pytest.raises(SystemExit, match=rf"Refusing llm_headline: .*{key}"):
        gnn_launch(s, tmp_path, FakeGnn(), monkeypatch)
    assert not any(tmp_path.iterdir())


def test_an_unfinished_gnn_checkpoint_refuses_before_a_b_c_bazt_run(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match=r"Refusing llm_headline: .*insider_pi__mmlu.*gnn_train"):
        gnn_launch(fakes.settings(*GNN), tmp_path, FakeGnn(finished=False), monkeypatch)
    assert not any(tmp_path.iterdir())


def test_a_block_without_the_gnn_arm_never_asks_the_gnn_feature(tmp_path, monkeypatch):
    gnn = FakeGnn(finished=False)
    s = fakes.settings(*LLM, ["llm.headline_arms=['A']"])
    assert gnn_launch(s, tmp_path, gnn, monkeypatch).batches and gnn.asked == []


# ================================================================================================ calibration.mode
def test_a_calibration_block_needs_calibration_mode_set_to_a_known_value(tmp_path):
    with pytest.raises(SystemExit, match=r"Refusing mock_calibration: .*calibration\.mode"):
        calibrate_then(tmp_path, without(fakes.settings(*MOCK_B), "calibration.mode"))
    with pytest.raises(SystemExit, match=r"Refusing mock_calibration: calibration\.mode must be one of.*'sideways'"):
        calibrate_then(tmp_path, fakes.settings(*MOCK_B, ["calibration.mode='sideways'"]))
    assert not any(tmp_path.iterdir())


def test_an_evaluation_block_does_not_need_calibration_mode(tmp_path):
    assert launch(without(fakes.settings(*MOCK_A), "calibration.mode"), "mock_frontier", tmp_path).batches


# ================================================================================================ oracle_alloc
ORACLE = [fakes.RUNTIME_MOCK, fakes.TINY_MOCK, ["mock.uncapped_arms=['random_alloc', 'oracle_alloc']",
                                                "mock.capped_arms=[]", "mock.allocation_arms=[]",
                                                "mock.calibrated_arms=['random_alloc']"]]


def test_oracle_alloc_acts_at_the_audit_score_threshold_random_alloc_was_calibrated_to(tmp_path):
    s = fakes.settings(*ORACLE)
    calibrate_then(tmp_path, s, fakes.FakePlatform("mock", calibrate_result={"audit_score": 0.7}))
    plat = launch(s, "mock_frontier", tmp_path)
    sent = [u for _, units in plat.batches for u in units]
    assert {u["arm"] for u in sent} == {"random_alloc", "oracle_alloc"}
    assert all(u["thresholds"] == {"audit_score": 0.7} for u in sent)


def test_oracle_alloc_is_refused_when_random_allocs_threshold_is_missing(tmp_path):
    s = fakes.settings(*ORACLE)
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: oracle_alloc needs the calibrated audit_score of random_alloc"):
        launch(s, "mock_frontier", tmp_path)
    assert not any(tmp_path.iterdir())
    calibrate_then(tmp_path, s, fakes.FakePlatform("mock", calibrate_result={"deny_score": 0.7}))
    with pytest.raises(SystemExit, match=r"oracle_alloc needs the calibrated audit_score"):
        launch(s, "mock_frontier", tmp_path)


# ===================================================================================== adaptive refusals as values
@pytest.mark.parametrize("failing", ["search_episodes", "run_search"])
def test_an_adaptive_search_that_refuses_with_a_value_error_is_a_refusal(tmp_path, failing):
    class Overlap(fakes.FakeAdaptive):
        pass

    nothing_written = failing == "search_episodes"

    def refuse(self, *args, **kwargs):
        raise ValueError("search seeds [1, 2] overlap mock.main_seeds")
    setattr(Overlap, failing, refuse)
    s = fakes.settings(*CAL_B)
    calibrate_then(tmp_path, s)
    with pytest.raises(SystemExit, match=r"Refusing mock_adaptive: search seeds \[1, 2\] overlap"):
        rx.run_block(s, "mock_adaptive", False, wiring=fakes.wiring({"mock": fakes.FakePlatform("mock")},
                                                                    adaptive=Overlap()), results_root=tmp_path)
    assert not nothing_written or not block_dir(tmp_path, "mock_adaptive").exists()


def test_the_recalibrated_gnn_runs_on_its_own_calibrated_threshold(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(*GNN, ["llm.headline_arms=['GNN_recalibrated']", "llm.calibrated_arms=['GNN_recalibrated']"])
    launch(s, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    plat = gnn_launch(s, tmp_path, FakeGnn(), monkeypatch)
    assert plat.batches and all(u["thresholds"] == {"deny_score": fakes.threshold_of(rx.cell_key(u))}
                                for _, (u,) in plat.batches)
