"""The pinned G-Safeguard checkout and our patch set.

`prepare` builds a derived copy of the upstream Python files (LF line endings, whatever the checkout uses), applies
patches/*.patch to it with `git apply` (`--check` first), and returns its path. The submodule working tree is never
modified and nothing of upstream is copied into git. `load_family` imports one attack family's patched modules under
private names, so PI/TA/MA modules with the same file names (agents.py, utils.py) do not collide.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

REPO = Path(__file__).resolve().parents[4]
SUBMODULE = REPO / "vendor" / "G-safeguard"
PATCH_DIR = Path(__file__).resolve().parents[1] / "patches"
DEFAULT_DEST = REPO / "results" / "experiments" / "_g-safeguard-patched"
PINNED_COMMIT = "890c99f1cbc864e9ff0c85859619a14f42bc9cab"
PATCHES = ("0001-seeds.patch", "0002-imports-and-paths.patch", "0003-defender-hook.patch",
           "0004-delivery-masks.patch", "0005-sender-outbox.patch", "0006-external-context.patch")
CODE_DIRS = ("PI", "TA", "MA", "scalability")
EXTRA_FILES = ("TA/tools.json",)
STAMP = ".harness"

# family -> (directory, agents module, graph-generator module that holds the system prompts)
FAMILIES = {
    "pi": ("PI", "agents", "gen_graph"),
    "pi_gsm8k": ("PI", "agents_for_gsm8k", "gen_graph_for_gsm8k"),
    "ta": ("TA", "agents", "gen_graph"),
    "ma": ("MA", "agents", "gen_graph"),
}
_SIBLINGS = ("agents", "agents_for_gsm8k", "main_defense_for_different_topology", "gen_graph", "gen_graph_for_gsm8k",
             "utils", "agent_prompts", "get_tool_attack_data", "gen_csqa", "gen_mmlu", "gen_gsm8k",
             "gen_memory_attack_data")


class HarnessError(RuntimeError):
    pass


def _git(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    env = None
    if cwd is not None:
        # The derived copy is not part of any repository. Inside one (results/ of this repository is gitignored)
        # git apply reports success and leaves every ignored file unpatched, so git must not discover a repository here.
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(Path(cwd).resolve().parent)}
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env, check=False)


def _patch_digest() -> str:
    h = hashlib.sha256()
    for name in PATCHES:
        h.update(name.encode() + b"\0" + (PATCH_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _stamp(commit: Optional[str]) -> str:
    return f"{commit or 'unpinned'}\n{_patch_digest()}\napply-checked\n"        # 2nd line of the version: copies built before the reverse check are rebuilt


def _verify_commit(source: Path, expect: Optional[str]) -> str:
    if expect is None:
        return "unpinned"
    head = _git("-C", str(source), "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != expect:
        raise HarnessError(f"{source} must be the G-Safeguard checkout at {expect[:7]} (D33), found "
                           f"{head.stdout.strip()[:7] or head.stderr.strip()!r}; run `git submodule update --init`")
    return expect


def _copy_code(source: Path, into: Path) -> None:
    """The upstream .py files (plus TA/tools.json), CRLF read as LF."""
    for d in CODE_DIRS:
        for path in (source / d).glob("*.py"):
            (into / d).mkdir(parents=True, exist_ok=True)
            (into / d / path.name).write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    for rel in EXTRA_FILES:
        shutil.copyfile(source / rel, into / rel)


def _apply_all(workdir: Path) -> None:
    for name in PATCHES:
        patch = (PATCH_DIR / name).resolve()
        for step in (("--check",), ()):
            # autocrlf off: a user's core.autocrlf=true would rewrite every patched file with CRLF
            done = _git("-c", "core.autocrlf=false", "apply", *step, "-p1", "--whitespace=nowarn", str(patch),
                        cwd=workdir)
            if done.returncode != 0:
                raise HarnessError(f"{name} does not apply{' (git apply --check)' if step else ''}: "
                                   f"{done.stderr.strip()}")
        undo = _git("-c", "core.autocrlf=false", "apply", "--check", "--reverse", "-p1", "--whitespace=nowarn", str(patch),
                    cwd=workdir)
        if undo.returncode != 0:                      # applied means: the patch can now be reversed
            raise HarnessError(f"{name} reported success but left the files unpatched: {undo.stderr.strip()}")


def prepare(dest: Optional[Path] = None, source: Optional[Path] = None,
            expect_commit: Optional[str] = PINNED_COMMIT) -> Path:
    """The patched, importable copy of the pinned checkout. Reused when its stamp (commit + patch set) matches,
    rebuilt otherwise. `expect_commit=None` skips the commit check (a copy that is not a git checkout)."""
    dest, source = Path(dest or DEFAULT_DEST), Path(source or SUBMODULE)
    commit = _verify_commit(source, expect_commit)
    stamp = _stamp(commit)
    if (dest / STAMP).is_file() and (dest / STAMP).read_text(encoding="utf-8") == stamp:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=dest.name + ".", dir=dest.parent))
    try:
        _copy_code(source, build)
        _apply_all(build)
        (build / STAMP).write_text(stamp, encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        build.rename(dest)
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise
    return dest


def load_family(root: Path, family: str) -> SimpleNamespace:
    """`.agents` (module with Agent, AgentGraphWithDefense, llm_invoke, allm_invoke), `.main` (defense_communication),
    `.graph` (generate_agent_graph_dataset: the family's system prompts) and `.utils` of one family."""
    directory, agents_name, graph_name = FAMILIES[family]
    path = str(Path(root) / directory)
    saved = {n: sys.modules.pop(n) for n in _SIBLINGS if n in sys.modules}
    sys.path.insert(0, path)
    stub_einops = importlib.util.find_spec("einops") is None          # imported, never called, by some upstream modules
    if stub_einops:
        sys.modules["einops"] = SimpleNamespace(rearrange=None)
    try:
        agents = importlib.import_module(agents_name)
        main = importlib.import_module("main_defense_for_different_topology")
        graph = importlib.import_module(graph_name)
        utils = importlib.import_module("utils")
        extra = {}
        if family == "ta":
            extra["get_tool_attack_data"] = importlib.import_module("get_tool_attack_data")
    finally:
        sys.path.remove(path)
        if stub_einops:
            sys.modules.pop("einops", None)
        for n in _SIBLINGS:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return SimpleNamespace(family=family, root=Path(root), agents=agents, main=main, graph=graph, utils=utils, **extra)
