#!/usr/bin/env python3
"""Verify that this machine can reproduce a run: Python, pinned packages, lock file, torch on CPU, secrets present.

    python scripts/check_env.py                    # exit 0 when everything matches, 1 on any mismatch
    python scripts/check_env.py --require-secrets  # a missing HF_TOKEN / WANDB_API_KEY / OPENROUTER_API_KEY fails too
    python scripts/check_env.py --threads 8        # torch must use exactly 8 threads

Checks, in order:
  python     3.11, the version pyproject.toml requires
  packages   every dependency pinned in pyproject.toml ([project] and [dependency-groups]) is installed at exactly its
             pin (a local label such as torch's "+cpu" is ignored); a dependency that is not an exact `==` pin fails
  lock       uv.lock resolves every pin to the same version (so a hand edit of pyproject.toml without `uv lock` fails)
  torch      a CPU build, CPU as the default device, and the thread count (printed; results are only bitwise
             reproducible for a fixed thread count, so the manifest records it too)
  secrets    HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY are set, from the environment or from .env. Only presence is
             reported, never a value. A missing one is a warning unless --require-secrets is given.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ("HF_TOKEN", "WANDB_API_KEY", "OPENROUTER_API_KEY")
EXACT_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([A-Za-z0-9][A-Za-z0-9._!+-]*)\s*(?:;.*)?$")
MARK = {"ok": "ok", "warn": "warn", "fail": "FAIL"}


@dataclass(frozen=True)
class Result:
    status: str          # ok | warn | fail
    name: str
    detail: str = ""

    def __str__(self) -> str:
        return f"[{MARK[self.status]}] {self.name}" + (f": {self.detail}" if self.detail else "")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def base_version(version: str) -> str:
    """The release without a local label: 2.14.0+cpu -> 2.14.0."""
    return version.split("+", 1)[0]


def same_release(installed: str, pin: str) -> bool:
    return base_version(installed) == pin


# ------------------------------------------------------------------------------------------------- the pins
def parse_pins(pyproject: Mapping) -> tuple[dict[str, str], list[str]]:
    """({normalised name: version}, problems) from [project].dependencies and every [dependency-groups] list."""
    specs = list(pyproject.get("project", {}).get("dependencies", []))
    for group in pyproject.get("dependency-groups", {}).values():
        specs += [item for item in group if isinstance(item, str)]
    pins: dict[str, str] = {}
    problems: list[str] = []
    for spec in specs:
        match = EXACT_PIN.match(spec)
        if match:
            pins[normalize(match.group(1))] = match.group(2)
        else:
            problems.append(f"not an exact pin: {spec!r}")
    return pins, problems


def lock_versions(lock: Mapping) -> dict[str, set[str]]:
    """{normalised name: release versions} from uv.lock; torch has one entry per platform marker."""
    found: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        found.setdefault(normalize(package["name"]), set()).add(base_version(package["version"]))
    return found


# ---------------------------------------------------------------------------------------------- the checks
def check_python(version_info: tuple) -> Result:
    found = ".".join(map(str, version_info[:3]))
    if tuple(version_info[:2]) == (3, 11):
        return Result("ok", "python", found)
    return Result("fail", "python", f"{found}, pyproject.toml requires 3.11")


def check_packages(pins: Mapping[str, str], installed: Callable[[str], str | None]) -> list[Result]:
    results = []
    for name, pin in sorted(pins.items()):
        found = installed(name)
        if found is None:
            results.append(Result("fail", name, f"not installed, pinned {pin}"))
        elif same_release(found, pin):
            results.append(Result("ok", name, found))
        else:
            results.append(Result("fail", name, f"installed {found}, pinned {pin}"))
    return results


def check_lock(pins: Mapping[str, str], locked: Mapping[str, set[str]]) -> list[Result]:
    results = []
    for name, pin in sorted(pins.items()):
        versions = locked.get(name)
        if versions is None:
            results.append(Result("fail", f"uv.lock {name}", f"absent, pinned {pin}; run `uv lock`"))
        elif versions != {pin}:
            results.append(Result("fail", f"uv.lock {name}", f"locks {sorted(versions)}, pinned {pin}; run `uv lock`"))
    return results or [Result("ok", "uv.lock", f"agrees with the {len(pins)} pins")]


def check_torch(expected_threads: int | None) -> list[Result]:
    try:
        import torch
    except ImportError as error:
        return [Result("fail", "torch", f"cannot be imported: {error}")]
    results = []
    cuda = torch.version.cuda
    results.append(Result("fail" if cuda else "ok", "torch build",
                          f"CUDA {cuda} build, install the CPU wheel with `uv sync`" if cuda else "CPU"))
    device = torch.get_default_device().type
    placed = torch.zeros(1).device.type
    results.append(Result("ok" if device == placed == "cpu" else "fail", "torch device",
                          f"default {device}, new tensors on {placed}"))
    threads = torch.get_num_threads()
    detail = f"{threads} (interop {torch.get_num_interop_threads()})"
    if expected_threads is not None and threads != expected_threads:
        results.append(Result("fail", "torch threads", f"{detail}, expected {expected_threads}"))
    else:
        results.append(Result("ok", "torch threads", detail))
    return results


def read_environment(dotenv: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """The variables the adapters will see: .env fills gaps, the real environment wins (override=False)."""
    values: dict[str, str] = {}
    if dotenv.is_file():
        from dotenv import dotenv_values
        values.update({k: v for k, v in dotenv_values(dotenv).items() if v is not None})
    values.update(environ)
    return values


def check_secrets(environ: Mapping[str, str], require: bool) -> list[Result]:
    results = []
    for name in SECRETS:
        if environ.get(name):
            results.append(Result("ok", name, "set"))
        else:
            results.append(Result("fail" if require else "warn", name, "missing"))
    return results


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="directory with pyproject.toml, uv.lock and .env")
    parser.add_argument("--env-file", type=Path, default=None, help="dotenv file (default: <root>/.env)")
    parser.add_argument("--threads", type=int, default=None, help="require exactly this many torch threads")
    parser.add_argument("--require-secrets", action="store_true", help="a missing secret is a failure")
    args = parser.parse_args(argv)

    pyproject = tomllib.loads((args.root / "pyproject.toml").read_text(encoding="utf-8"))
    pins, problems = parse_pins(pyproject)
    results = [check_python(sys.version_info), *(Result("fail", "pyproject.toml", p) for p in problems)]
    results += check_packages(pins, installed_version)
    lock = args.root / "uv.lock"
    if lock.is_file():
        results += check_lock(pins, lock_versions(tomllib.loads(lock.read_text(encoding="utf-8"))))
    else:
        results.append(Result("fail", "uv.lock", "missing; run `uv lock`"))
    results += check_torch(args.threads)
    environment = read_environment(args.env_file or args.root / ".env", os.environ)
    results += check_secrets(environment, args.require_secrets)

    for result in results:
        print(result)
    failures = sum(r.status == "fail" for r in results)
    warnings = sum(r.status == "warn" for r in results)
    print(f"environment check: {failures} failure{'' if failures == 1 else 's'}, "
          f"{warnings} warning{'' if warnings == 1 else 's'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
