"""The runner and the composition root obey the import rules of docs/ARCHITECTURE_EXPERIMENTS.md."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_the_runner_and_the_composition_root_pass_the_import_gate():
    run = subprocess.run([sys.executable, "-m", "bazt_lab.tools.check_architecture"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    assert "architecture gate:" in run.stdout, run.stderr
    mine = [line for line in run.stdout.splitlines() if line.startswith(("bazt_lab/app.py", "bazt_lab/features/runner"))]
    assert mine == []
