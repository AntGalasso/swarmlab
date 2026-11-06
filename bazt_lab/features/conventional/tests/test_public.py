"""What the runner sees of the conventional arms: public.ARMS, and the lane's place in the import gate."""
from pathlib import Path

from bazt_lab.core import public as core
from bazt_lab.features.conventional import public
from bazt_lab.tools import check_architecture as gate

REPO = Path(__file__).resolve().parents[4]


def test_arms_are_a_b_and_c_with_the_flags_the_runner_reads():
    assert [spec.name for spec in public.ARMS] == ["A", "B", "C"]
    assert [spec.calibrated for spec in public.ARMS] == [False, True, True]       # settings: mock.calibrated_arms
    for spec in public.ARMS:
        assert isinstance(spec, core.ArmSpec)
        assert spec.platforms == ("mock", "llm") and spec.needs_truth is False


def test_each_factory_builds_a_fresh_defender_of_its_name():
    for spec in public.ARMS:
        first, second = spec.factory(), spec.factory()
        assert first is not second and isinstance(first, core.Defender) and first.name == spec.name


def test_the_lane_has_no_import_gate_violation():
    mine = [str(v) for v in gate.check(REPO) if v.path.startswith("bazt_lab/features/conventional")]
    assert mine == []
