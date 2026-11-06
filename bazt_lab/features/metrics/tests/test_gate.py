"""The metrics feature obeys the layout and import rules of docs/ARCHITECTURE_EXPERIMENTS.md (the repo-wide gate is
also run by tests/test_architecture_bazt_lab.py, but it fails for every feature that is unfinished)."""
from bazt_lab.tools import check_architecture as gate

FEATURE = "bazt_lab/features/metrics"


def test_the_metrics_feature_passes_the_import_and_layout_gate():
    found = [str(v) for v in gate.check() if v.path.startswith(FEATURE)]
    assert found == []
