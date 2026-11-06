#!/usr/bin/env python3
"""Static checks for our code: ruff, byte-compilation and the import gate. Every step runs; exit 1 if any failed.

    python scripts/check_static.py                 # all three
    python scripts/check_static.py --fix           # let ruff apply its safe fixes first
    python scripts/check_static.py --only ruff     # one step: ruff, compileall or "import gate"

ruff takes no path: ruff.toml limits it to bazt_lab/, scripts/ and tests/test_architecture_bazt_lab.py, so the upstream
files are never linted. The import gate is bazt_lab/tools/check_architecture.py (docs/ARCHITECTURE_EXPERIMENTS.md).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("ruff", "compileall", "import gate")


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]


def steps(fix: bool, only: str | None) -> list[Step]:
    if only is not None and only not in NAMES:
        raise ValueError(f"unknown step {only!r}; choose one of {', '.join(NAMES)}")
    every = [
        Step("ruff", [sys.executable, "-m", "ruff", "check", *(["--fix"] if fix else [])]),
        Step("compileall", [sys.executable, "-m", "compileall", "-q", "bazt_lab", "scripts"]),
        Step("import gate", [sys.executable, "-m", "bazt_lab.tools.check_architecture"]),
    ]
    return [s for s in every if only in (None, s.name)]


def run(to_run: list[Step], cwd: Path = ROOT) -> int:
    lines: list[str] = []
    for step in to_run:
        print(f"\n== {step.name}: {' '.join(step.argv[1:])}", flush=True)
        try:
            code = subprocess.run(step.argv, cwd=cwd, check=False).returncode
        except OSError as error:
            print(f"cannot start {step.argv[0]}: {error}")
            code = 127
        lines.append(f"ok    {step.name}" if code == 0 else f"FAIL  {step.name} (exit {code})")
    print("\n" + "\n".join(lines))
    failed = sum(line.startswith("FAIL") for line in lines)
    print(f"static checks: {failed} failed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fix", action="store_true", help="let ruff apply its safe fixes")
    parser.add_argument("--only", choices=NAMES, default=None, help="run one step")
    args = parser.parse_args(argv)
    return run(steps(args.fix, args.only))


if __name__ == "__main__":
    sys.exit(main())
