"""llm_adaptive_check end to end through the real runner, offline: a tiny mock_adaptive search on the real mock engine,
then the check on the LLM platform's fake API (llm.api = "fake": FakeBackend, FakeJudge, synthetic items; no network),
against the calibrated arm B. The kickoff guard stays: the test moves the kickoff into the past, as the runner tests do."""
import json

from bazt_lab.core import public as core
from bazt_lab.features.adaptive import public as adaptive
from bazt_lab.features.conventional import public as conventional
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.mock_engine import public as mock_engine
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_launch import past_kickoff

SEARCH = ["mock.adaptive.search_seeds=[9101, 9102]", "mock.adaptive.insider_sets='contiguous'",
          "mock.adaptive.start_fractions=[0.0, 0.5]", "mock.adaptive.rates=[0.5, 1.0]",
          "mock.adaptive.detectabilities=[0.5, 1.0]", "mock.adaptive.evasion_cost=0.5",
          "mock.adaptive.max_candidates=1000", "mock.adaptive.knowledge.gray_box='cell'",
          "mock.adaptive.arms=['B']", "mock.adaptive.cells=[[10, 10]]", "mock.adaptive.conditions=['gray_box']"]
CHECK = ["llm.api='fake'", "llm.agents=3", "llm.backbones=['fake/bb']", "llm.sanitize_modes=['redaction']",
         "llm.conversations_per_cell=4", "llm.calibration_conversations_per_cell=2", "calibration.mode='open_loop'",
         "llm.adaptive_check.arms=['B']", "llm.adaptive_check.families=['infection']",
         "llm.adaptive_check.condition='gray_box'", "llm.adaptive_check.tasks=['mmlu']", "llm.adaptive_check.mock_cell=[10, 10]",
         "llm.adaptive_check.mock_topology.hub_star='chain'", "llm.adaptive_check.start_clock='conversation'"]


def wiring():
    arms = {a.name: a for a in conventional.ARMS}
    platforms = {"mock": mock_engine.MockPlatform(), "llm": harness.LLMPlatform()}
    return rx.Wiring(arm=lambda name: arms[name], platform=lambda name: platforms[name], tracker=lambda s: None,
                     adaptive=lambda: adaptive)


def rows_of(block_dir):
    files, uncommitted = core.committed_row_files(block_dir)
    assert uncommitted == 0
    return [json.loads(line) for path in files for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_the_llm_adaptive_check_replays_a_real_mock_search_on_the_fake_api(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(fakes.RUNTIME_MOCK, fakes.TINY_MOCK, fakes.CALIBRATED_B, fakes.RUNTIME_LLM, fakes.TINY_LLM,
                       SEARCH, CHECK)
    w = wiring()
    for block in ("mock_calibration", "mock_adaptive", "llm_calibration", "llm_adaptive_check"):
        rx.run_block(s, block, block.startswith("llm"), wiring=w, results_root=tmp_path)
    table = adaptive.load_best_responses(tmp_path / "sprint" / "mock_adaptive", s)
    out = tmp_path / "sprint" / "llm_adaptive_check"
    pin = json.loads((out / "best_responses.json").read_text(encoding="utf-8"))
    ((cell, params),) = pin["entries"].items()
    chosen = table.choice({**json.loads(cell), "seed": None, "sink": None}).attack_params
    assert params == {"start_frac": float(chosen.get("start_frac", 0.0)), "rate": float(chosen.get("rate", 1.0))}
    rows = rows_of(out)
    units = rx.units_llm_adaptive_check(s, True)
    assert len(rows) == len(units) * units[0]["conversations"]
    start = int(params["start_frac"] * 4)                          # 4 rounds per conversation, conversation clock
    for r in rows:
        assert r["arm"] == "B" and r["condition"] == "gray_box" and r["attack"] == "infection"
        assert r["attack_schedule"] == {"clock": "conversation", "start_frac": params["start_frac"],
                                        "start_round": start, "rate": params["rate"]}
        assert r["meta_attack_rounds"] == [i >= start for i in range(4)]
        assert r["empty_replies"] == 0
