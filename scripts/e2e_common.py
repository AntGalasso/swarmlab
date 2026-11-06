"""Shared harness of the end-to-end smoke runs (scripts/e2e_mock.py, scripts/e2e_llm_fake.py).

A smoke run drives the real launcher (run_experiments.py) as a subprocess on a tiny grid, in a temporary results
directory, and checks what it leaves on disk:

  1. --list and --dry-run work and report the unit count the grid must have
  2. each calibration block writes thresholds.json with one record per cell
  3. each evaluation block is killed once, part-way (the process is killed, not asked to stop), then launched again;
     the second launch must finish the block, keep every finished unit file byte for byte, continue an unfinished
     file after its last saved row, and leave no torn line, no duplicate row, no stray temporary file. On a platform
     with several rows per unit (LLM) the kill must leave a unit part-written, or the resume of a unit in the middle
     is not exercised: the block is killed again (at most KILL_ATTEMPTS launches) and otherwise the check fails
  4. a third launch of the finished block changes nothing
  5. every row has the metrics columns and obeys the invariants of check_row
  6. every launch left a manifest that records the test values
  7. the analysis runs on the output and read every committed row of the blocks it reads

Rows are read the way the analysis reads them: the committed rows files of `core.committed_row_files` (<stem>.jsonl with
<stem>.done beside it, in units/ and in every worker shard units/w<k>/). A rows file without its done marker (a batch
cut before its done list, a unit still running or abandoned in a stale shard) is not a result.

The run-time variables of the spec (section 10) have no value in profiles.toml, so the smoke run supplies TEST VALUES
with --set. They exist only to let a tiny run start. They are not experiment settings, are never written to
profiles.toml, and say nothing about what the experiments should use.

The child process gets empty HF_TOKEN, WANDB_API_KEY and OPENROUTER_API_KEY (present but empty, so python-dotenv with
override=False cannot refill them from .env): a smoke run cannot make an authenticated call.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "experiments" / "profiles.toml"
SECRETS = ("HF_TOKEN", "WANDB_API_KEY", "OPENROUTER_API_KEY")
ANALYSIS_ENTRY = "bazt_lab.features.analysis.public:analyze"

# E2E TEST VALUES for the run-time variables of [runtime] in profiles.toml. Only to start a tiny smoke run.
RUNTIME_TEST_VALUES: dict[str, object] = {
    "n_insiders": 2, "rate_limit_per_identity": 1, "horizon_constant": 2.0, "eps_correct": 5.0,
    "recovery_margin": 0.05, "ni_margin_accuracy": 0.01, "rank_agreement_threshold": 0.5,
    "random_edge_probability": 0.4, "ba_m": 2, "ws_k": 2, "ws_rewire_p": 0.2,
    "llm_seeds": [1, 2], "llm_calibration_seeds": [101, 102],
    "sanitize_success": 0.9, "insider_bias": 5.0, "ta_detectability": 1.0, "ma_detectability": 1.0,
    "aitm_rewrite_strength": 0.8,
}
# E2E TEST VALUES for the open decisions of the arms and of the mock auditor (tables outside [runtime] whose keys are
# commented out in profiles.toml). Same rule: only to start a tiny smoke run, never written to profiles.toml.
BIG = 1e9           # an upper bin edge standing for infinity (TOML written by json has no inf)
ARM_TEST_VALUES: dict[str, object] = {
    "conventional.c.judge_score_bins": {"mock": [[0.0, 0.0], [2.0, 4.0], [BIG, 12.0]],
                                        "llm": [[-5.0, 0.0], [-1.0, 4.0], [0.0, 12.0]]},
    "conventional.c.presence_window_rounds": 2,
    "mock.auditor.binormal_method": "shared_threshold",
    "bazt.katz_decay": 0.1, "bazt.eigen_iterations": 5, "bazt.sink_weight": 2.0, "bazt.utility_weight": 0.5,
    "bazt.flag_logodds": 2.0, "bazt.clear_logodds": 1.0, "bazt.memory_decay": 0.5, "bazt.prior": 0.1,
    "bazt.inspect_conclusive": 0.9,
    "bazt.utility_loss": {"sanitize": 0.1, "delay": 0.1, "edge_quarantine": 0.2, "rate_limit": 0.1,
                          "privilege_degradation": 0.2, "isolation": 0.5},
    "bazt_upstream.audit_verdict_threshold": 0.0, "bazt_upstream.verdict_probability": 0.98,
    "llm.judge.provider": "fake-provider", "llm.judge.top_logprobs": 20, "llm.judge.floor_probability": 1e-6,
    # compute.dtype is not set here: the e2e runs the dtype of profiles.toml, the one every real run uses (D117:
    # float32). check_gpu.py sets it per combination and keeps CPU float64 as its reference.
    "compute.device": "cpu", "compute.compile": False, "compute.deterministic": True, "compute.workers": 1,
    "tracking.flush_rows": 3600, "sync.enabled": False, "sync.repo_id": "e2e-test/never-uploaded", "gnn_train.keep_checkpoints": 2,
    "llm.providers": {"openai/gpt-4o-mini": "fake-provider"},
    "gnn_train.split": "sequential", "gnn_train.seed": 7,
    "gnn.recalibrate": False, "gnn.embedding.fake": True, "gnn.carry_flags": True, "gnn.checkpoint_family": "insider_pi__mmlu",
    "gnn.fallback_family": "pi_same_task", "calibration.mode": "open_loop",
    "llm.sanitize_modes": ["redaction"], "llm.ma_correctness": ["rule"], "llm.ma_judge_model": "gpt-4o-mini",
    "mock.adaptive.search_seeds": [9101, 9102], "mock.adaptive.insider_sets": "contiguous",
    "mock.adaptive.start_fractions": [0.0, 0.5], "mock.adaptive.rates": [0.5, 1.0],
    "mock.adaptive.detectabilities": [0.5, 1.0], "mock.adaptive.evasion_cost": 0.5, "mock.adaptive.max_candidates": 1000,
    "mock.adaptive.knowledge": {"gray_box": "cell", "white_box": "seed", "embedding_evasion": "cell",
                                "attack_rate": "cell"},
    "mock.legacy.confirm_steps": 2, "mock.legacy.lam": 1.0, "mock.legacy.gamma": 0.7, "mock.legacy.dmax": 3,
    "mock.legacy.tau_san": 0.05, "mock.legacy.tau_central": 0.75, "mock.legacy.tau_confirm": 0.85,
    "mock.legacy.random_steps": 3,
    "runtime.llm_sink_positions": {"random_directed": [0], "hub_star": [0], "barabasi_albert": [0],
                                   "watts_strogatz": [0]},
}
# The open [analysis] keys: test values for the analysis step only (Plan.analysis_sets). The block lists name the blocks
# of both platforms; a platform whose block was not run reports no_data.
ANALYSIS_TEST_VALUES: dict[str, object] = {
    "analysis.challengers": ["BAZT"], "analysis.containment_metrics": ["agents_reached_auc"],
    "analysis.efficiency_tolerance": 0.0, "analysis.adaptive_contrast": "challenger_degrades_less_than_c",
    "analysis.rank_agreement_statistic": "spearman",
    "analysis.rank_agreement_arms": [["A", "A"], ["B", "B"], ["C", "C"], ["BAZT", "BAZT"]],
    "analysis.utility_endpoint": "accuracy_attack",
    "analysis.split_by": {"mock": [], "llm": []},
    "analysis.extra_metrics": {"protected_sink_harm": ["binary", "attacked"]},
    "analysis.topology_classes": {"mock": {"heterogeneous": ["star_hub"], "homogeneous": ["chain"]},
                                  "llm": {"heterogeneous": ["hub_star"], "homogeneous": ["random_directed"]}},
    "analysis.blocks": {
        "budget": {"mock": ["mock_frontier"], "llm": ["llm_capped"]},
        "efficiency": {"mock": ["mock_frontier"], "llm": []},
        "topology": {"mock": ["mock_frontier"], "llm": ["llm_headline"]},
        "adaptive": {"mock": [], "llm": []},
        "adaptive_reference": {"mock": [], "llm": []},
        "ladder": {"mock": ["mock_frontier"], "llm": ["llm_headline"]},
        "ablations": {"mock": [], "llm": []},
        "rank": {"mock": ["mock_frontier"], "llm": ["llm_headline"]},
        "containment": {"mock": ["mock_frontier"], "llm": ["llm_capped", "llm_headline"]},     # A, the reference, runs in llm_headline
    },
}
# Documented open keys the smoke runs deliberately leave unset: they enable blocks the smoke runs do not run, are set
# by the plan itself, or belong to the owner (tracking).
NOT_SET_BY_E2E = frozenset({
    "llm.backbones", "llm.calibration_conversations_per_cell", "llm.api", "gnn_train.backbone",
    "llm.adaptive_check.arms", "llm.adaptive_check.families", "llm.adaptive_check.condition", "llm.adaptive_check.tasks",
    "llm.adaptive_check.mock_cell", "llm.adaptive_check.mock_topology", "llm.adaptive_check.start_clock",
    "llm.adaptive_check.mock_arm",
    "mock.epidemiology.arms", "mock.epidemiology.cells", "mock.ablations.arms", "mock.ablations.cells",
    "mock.adaptive.arms", "mock.adaptive.cells", "mock.presets_robustness.arms", "mock.presets_robustness.cells",
    "mock.costs_robustness.arms", "mock.costs_robustness.cells", "mock.legacy.arms", "mock.legacy.cells",
    "costs.robustness.deny", "tracking.enabled", "tracking.entity",
})
RATIOS = ("asr_final", "accuracy_clean", "accuracy_attack", "nrp", "fpr_message", "fpr_agent", "benign_delivery_rate",
          "agents_reached_auc", "protected_sink_harm", "sink_exposed", "targeted_asr", "contained")
FINITE_NON_NEGATIVE = ("spend_audit_total", "spend_intervention_total", "spend_total", "violations", "exhausted_step")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------------------------------ test values
def documented_runtime_keys(profiles_text: str) -> list[str]:
    """The run-time variables profiles.toml documents as commented-out keys of its [runtime] table."""
    block = re.search(r"^\[runtime\]\s*$(.*?)(?=^\[)", profiles_text, re.S | re.M)
    return re.findall(r"^#\s*(\w+)\s*=", block.group(1), re.M) if block else []


def missing_test_values(profiles_text: str) -> list[str]:
    return [k for k in documented_runtime_keys(profiles_text) if k not in RUNTIME_TEST_VALUES]


def documented_arm_keys(profiles_text: str) -> list[str]:
    """Dotted keys that profiles.toml documents as commented-out entries, in any table but [runtime]. A commented
    table header (`# [analysis.split_by]`) switches the table like a real one."""
    found, table = [], ""
    for line in profiles_text.splitlines():
        header = re.match(r'^#?\s*\[([A-Za-z_][\w."|-]*)\]', line)
        if header:
            table = header.group(1)
            continue
        key = re.match(r"^#\s*([A-Za-z_]\w*)\s*=", line)
        if key and table and table != "runtime" and table.split(".")[0] != "estimates":     # sizes, not decisions
            found.append(f"{table}.{key.group(1)}")
    return found


def documented_options(profiles_text: str) -> dict[str, set[str]]:
    """{dotted key: the quoted options its comment lists} for commented keys whose comment names two or more quoted
    words (`# rank_agreement_statistic = "..."   # OPEN: "spearman" or "kendall"`). Indented comment lines under a key
    belong to it."""
    options: dict[str, set[str]] = {}
    table, key = "", None
    for line in profiles_text.splitlines():
        header = re.match(r"^#?\s*\[([A-Za-z_][\w.]*)\]", line)
        found = re.match(r"^#\s*([A-Za-z_]\w*)\s*=\s*(.*)$", line)
        if header:
            table, key = header.group(1), None
        elif found and table:
            key = f"{table}.{found.group(1)}"
            options[key] = set(re.findall(r'"([^"]+)"', found.group(2).partition("#")[2]))
        elif key and re.match(r"^\s+#", line):
            options[key] |= set(re.findall(r'"([^"]+)"', line))
        else:
            key = None
    return {k: v - {"..."} for k, v in options.items() if len(v - {"..."}) >= 2}


def invalid_choices(profiles_text: str) -> list[str]:
    """Test values that are text and not among the options their key's comment lists."""
    values = {**ARM_TEST_VALUES, **ANALYSIS_TEST_VALUES}
    return [f"{key}={values[key]!r} is not one of {sorted(options)}" for key, options in documented_options(
        profiles_text).items() if isinstance(values.get(key), str) and values[key] not in options]


def missing_arm_values(profiles_text: str) -> list[str]:
    values = {**ARM_TEST_VALUES, **ANALYSIS_TEST_VALUES}
    covered: set[str] = set()

    def add(prefix: str, value: object) -> None:
        covered.add(prefix)
        if isinstance(value, dict):
            for leaf, inner in value.items():
                add(f"{prefix}.{leaf}", inner)

    for key, value in values.items():
        add(key, value)
    return [k for k in documented_arm_keys(profiles_text) if k not in covered and k not in NOT_SET_BY_E2E]


def tiny_mock_sets(agents: int, seeds: int) -> dict[str, object]:
    """A tiny mock grid. The LLM smoke run needs it too, because `run-all` also runs the mock blocks."""
    return {
        "mock.batch_size": 4,                       # several batches per group, so a kill lands between them
        "mock.topologies": ["chain", "star_hub"],
        "mock.families": ["infection", "insider_pi"],
        "mock.frontier_agents": [agents],
        "mock.epidemiology_agents": [agents],
        "mock.main_seeds": [1, seeds],
        "mock.tune_seeds": [9001, 9000 + seeds],
        # all 12 mock arms of the grid: the six uncapped, the three allocation arms and the three BAZT ablations
        "mock.uncapped_arms": ["A", "B", "C", "BAZT", "BAZT_upstream", "GNN_standin",
                               "BAZT_no_budget", "BAZT_no_topology", "BAZT_no_graduated"],
        "mock.capped_arms": ["B", "C", "BAZT", "BAZT_upstream"],
        "mock.allocation_arms": ["random_alloc", "oracle_alloc", "bra_greedy"],
        "mock.calibrated_arms": ["B", "C", "BAZT", "BAZT_upstream", "GNN_standin", "BAZT_no_budget",
                                 "BAZT_no_topology", "BAZT_no_graduated", "random_alloc", "bra_greedy"],
        "budgets.levels_percent": [10],
    }


def runtime_sets(**changes: object) -> dict[str, object]:
    """The E2E test values as `runtime.<name>` overrides; `changes` replace some (a name that is not one is refused)."""
    unknown = sorted(set(changes) - set(RUNTIME_TEST_VALUES))
    if unknown:
        raise KeyError(f"not a run-time variable of the spec: {', '.join(unknown)}")
    return {f"runtime.{name}": value for name, value in {**RUNTIME_TEST_VALUES, **changes}.items()}


def plan_runtime(sets: Mapping[str, object]) -> dict[str, object]:
    """The run-time variables a plan sets, by bare name (tables such as llm_sink_positions are not scalars here)."""
    return {key.removeprefix("runtime."): value for key, value in sets.items()
            if key.startswith("runtime.") and "." not in key.removeprefix("runtime.")}


def toml_value(value: object) -> str:
    """A value as TOML: JSON for numbers, strings, bools and lists; an inline table for a mapping."""
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items()) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(value)


def override_items(sets: Mapping[str, object]) -> list[str]:
    """`key=value` overrides, each value written as TOML."""
    return [f"{key}={toml_value(value)}" for key, value in sets.items()]


def set_args(sets: Mapping[str, object]) -> list[str]:
    """`--set key=value` command-line pairs."""
    return [arg for item in override_items(sets) for arg in ("--set", item)]


def child_env(base: Mapping[str, str], kickoff_override: str | None = None) -> dict[str, str]:
    env = dict(base)
    env.update(dict.fromkeys(SECRETS, ""))
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", WANDB_MODE="offline", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1")
    if kickoff_override:
        env["SWARMLAB_KICKOFF_UTC"] = kickoff_override
    return env


@contextlib.contextmanager
def offline_environment() -> Iterator[None]:
    """Run code that is not a child process (check_gpu.py runs the block runner in-process) in the environment of the
    e2e children: no secrets, HF and W&B offline. The caller's environment is restored on exit."""
    saved = dict(os.environ)
    os.environ.update(child_env(saved))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def columns() -> tuple[str, ...]:
    from bazt_lab.features.metrics import public as metrics
    return tuple(metrics.COLUMNS)


# ---------------------------------------------------------------------------------------------- the plan
@dataclass(frozen=True)
class Calibration:
    block: str
    units: int
    cells: int                 # records thresholds.json must hold


@dataclass(frozen=True)
class Evaluation:
    block: str
    units: int
    rows_per_unit: int | None  # None: the rows of a unit are not metrics rows and their number is not fixed (training)
    interrupt: bool = False    # kill once part-way, then resume
    metrics: bool = True       # check every row against the metrics columns and invariants


@dataclass
class Plan:
    name: str
    sets: dict[str, object]              # passed to the launcher with --set (the grid and the run-time test values)
    calibrations: list[Calibration]
    evaluations: list[Evaluation]
    run_flags: list[str] = field(default_factory=list)
    kickoff_override: str | None = None
    analysis_sets: dict[str, object] = field(default_factory=dict)   # test values only the analysis reads


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--launcher", type=Path, default=ROOT / "run_experiments.py",
                        help="the script that runs the experiments (default: run_experiments.py)")
    parser.add_argument("--profile", default=None, help="profile (default: active_profile of profiles.toml)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds allowed for one launch")
    parser.add_argument("--keep", action="store_true", help="keep the temporary results directory")
    parser.add_argument("--analysis-entry", default=ANALYSIS_ENTRY, metavar="MODULE:FUNCTION",
                        help=f"analysis function called as f(results_dir, settings) (default: {ANALYSIS_ENTRY})")


# --------------------------------------------------------------------------------------------- reporting
class Abort(Exception):
    """A stage cannot go on; the message says why."""


class GuardRefused(Exception):
    """The launcher refused to start an LLM block because of the kickoff guard."""


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def section(self, title: str) -> None:
        print(f"\n== {title}", flush=True)

    def ok(self, message: str) -> None:
        print(f"  ok    {message}", flush=True)

    def fail(self, message: str) -> None:
        print(f"  FAIL  {message}", flush=True)
        self.failures.append(message)

    def check(self, condition: bool, message: str) -> bool:
        (self.ok if condition else self.fail)(message)
        return condition

    def problems(self, label: str, problems: Sequence[str], limit: int = 6) -> bool:
        if not problems:
            self.ok(label)
            return True
        self.fail(f"{label}: {len(problems)} problem(s)")
        for problem in problems[:limit]:
            print(f"        {problem}")
        if len(problems) > limit:
            print(f"        ... and {len(problems) - limit} more")
        return False


# -------------------------------------------------------------------------------------- files on disk
IN_FLIGHT_GRACE = 0.5      # seconds to wait, once a kill is due, for a moment when a unit is half written
KILL_ATTEMPTS = 8          # launches killed at most to leave a unit part-written, before the resume path counts as
                           # not exercised
FREEZE_SETTLE = 0.02       # seconds for a suspension to take hold (thread suspension is asynchronous) before files are read
BATCH = "batch-"      # CheckpointStore.append_batch: units/batch-<hash>.jsonl holds the rows of a whole batch,
                      # units/batch-<hash>.done is a JSON list of the unit ids it commits
WINDOWS = sys.platform == "win32"


@dataclass(frozen=True)
class Progress:
    started: int               # rows files that hold at least one byte
    done: int                  # units marked done
    rows: int = field(default=0, compare=False)    # rows on disk (a batch is committed whole)


def done_units(units_dir: Path) -> int:
    """Units marked done: one per unit `.done` file, and the ids a batch `.done` lists."""
    count = 0
    for path in units_dir.rglob("*.done"):
        if path.stem.startswith(BATCH):
            with contextlib.suppress(OSError, json.JSONDecodeError):      # else being written: not committed yet
                count += len(json.loads(path.read_text(encoding="utf-8")))
        else:
            count += 1
    return count


def progress(units_dir: Path) -> Progress:
    """What a running launch has put on disk so far, committed or not (the kill decision needs the files in flight).
    Rows files may sit under units/ or, with workers, under units/w<k>/."""
    started = rows = 0
    if not units_dir.is_dir():
        return Progress(0, 0, 0)
    for path in units_dir.rglob("*.jsonl"):
        with contextlib.suppress(OSError):
            data = path.read_bytes()
            if data:
                started += 1
                rows += data.count(b"\n")
    return Progress(started, done_units(units_dir), rows)


def part_written(units_dir: Path, rows_per_unit: int) -> list[str]:
    """Unit rows files (not a batch) without their done marker that hold at least one complete row and fewer than
    `rows_per_unit`: a unit cut in the middle, the state a resume must continue after its last saved row. Paths are
    relative to `units_dir`."""
    found: list[str] = []
    if not units_dir.is_dir():
        return found
    for path in sorted(units_dir.rglob("*.jsonl")):
        if path.stem.startswith(BATCH) or path.with_suffix(".done").exists():
            continue
        with contextlib.suppress(OSError):
            if 0 < path.read_bytes().count(b"\n") < rows_per_unit:
                found.append(path.relative_to(units_dir).as_posix())
    return found


def committed_files(units_dir: Path) -> list[Path]:
    """The committed rows files of the block whose units/ folder is `units_dir`: `core.committed_row_files`, the reader
    the analysis uses (<stem>.jsonl with <stem>.done in the same folder, in units/ and every worker shard). An orphan
    batch, a unit still running and a unit abandoned in a stale shard are not results and are not read."""
    from bazt_lab.core import public as core
    try:
        return list(core.committed_row_files(units_dir.parent)[0])
    except FileNotFoundError:
        return []


def committed_rows(block_dir: Path) -> int:
    """Rows (JSON objects) in the committed rows files of a block folder; 0 when it has no units/ folder."""
    return sum(len(read_rows(path)[0]) for path in committed_files(block_dir / "units"))


def read_rows(path: Path) -> tuple[list[dict], int]:
    """(rows, number of lines that are not a JSON object). NaN is accepted: the rows hold it."""
    rows, torn = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
        else:
            if isinstance(value, dict):
                rows.append(value)
            else:
                torn += 1
    return rows, torn


def snapshot(units_dir: Path) -> dict[str, bytes]:
    """Every rows file on disk, committed or not: a restart must continue the unfinished ones, and a launch of a finished
    block must not touch any file."""
    return {p.relative_to(units_dir).as_posix(): p.read_bytes() for p in sorted(units_dir.rglob("*.jsonl"))}


def finished(units_dir: Path) -> set[str]:
    return {p.stem for p in units_dir.rglob("*.done")}


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_row(row: Mapping, expected_columns: Sequence[str], arm: str) -> list[str]:
    """Invariants of one metrics row (spec section 6, metrics/CONTRACT.md) that hold for every arm and platform."""
    problems = []
    absent = [c for c in expected_columns if c not in row]
    if absent:
        problems.append(f"missing columns: {', '.join(absent)}")
    if row.get("arm") != arm:
        problems.append(f"arm {row.get('arm')!r} in a file of arm {arm!r}")

    def value(name: str) -> float:
        v = row.get(name)
        return float(v) if is_number(v) else math.nan

    problems += [f"{name}={row[name]} outside [0, 1]" for name in RATIOS
                 if not math.isnan(value(name)) and not 0.0 <= value(name) <= 1.0]
    problems += [f"{name}={row[name]!r} is not a finite number >= 0" for name in FINITE_NON_NEGATIVE
                 if name in row and not (math.isfinite(value(name)) and value(name) >= 0)]
    ttc = value("time_to_containment")
    if not math.isnan(ttc) and ttc < 0:
        problems.append(f"time_to_containment={ttc} < 0")
    if "agents" in row and row.get("n_agents") != row["agents"]:
        problems.append(f"n_agents={row.get('n_agents')} but the unit has {row['agents']} agents")
    n = row.get("n_agents")
    if is_number(n) and is_number(row.get("sink")) and not 0 <= row["sink"] < n:
        problems.append(f"sink={row['sink']} outside [0, {n})")
    clean = row.get("attack_free")
    if "attack" in row and clean != (row["attack"] == "none"):
        problems.append(f"attack_free={clean} but attack={row['attack']!r}")
    if clean is True:
        if value("n_attackers") != 0:
            problems.append(f"an attack-free row has n_attackers={row.get('n_attackers')!r}")
        if not math.isnan(value("asr_final")):
            problems.append("an attack-free row reports asr_final")
        if math.isnan(value("accuracy_clean")):
            problems.append("an attack-free row has no accuracy_clean")
    elif clean is False:
        if math.isnan(value("accuracy_attack")):
            problems.append("an attack row has no accuracy_attack")
        if not math.isnan(value("accuracy_clean")):
            problems.append("an attack row reports accuracy_clean")
    if arm == "A" and value("spend_total") != 0:
        problems.append(f"arm A costs nothing but spend_total={row.get('spend_total')}")
    uncapped = row.get("audit") == "uncapped" and row.get("intervention") == "uncapped"
    if uncapped and (value("violations") != 0 or value("exhausted_step") != 0):
        problems.append("an uncapped unit reports a budget violation or exhaustion")
    return problems


def check_rows(units_dir: Path, expected_columns: Sequence[str]) -> list[str]:
    """check_row on every row of every committed rows file. The arm is the part of a unit file's name before its digest;
    the rows of a batch file are of several arms, so each row names its own."""
    problems = []
    for path in committed_files(units_dir):
        arm = None if path.stem.startswith(BATCH) else path.stem.rsplit("-", 1)[0]
        for row in read_rows(path)[0]:
            problems += [f"{path.name}: {p}" for p in check_row(row, expected_columns, arm or row.get("arm"))]
    return problems


# What the arms of one scenario must agree on: they all face the same graph, attackers and sink (common random
# numbers, spec section 8: arm-independent draws never depend on the arm).
SCENARIO = ("platform", "topology", "attack", "agents", "seed", "sink", "backbone", "task", "condition", "preset",
            "cost_condition")
GROUND_TRUTH = ("n_attackers", "n_agents", "attack_free")


def check_scenarios(rows: Iterable[Mapping]) -> list[str]:
    """One problem per ground-truth column whose value differs between two rows of the same scenario."""
    seen: dict[tuple, dict[str, tuple]] = {}
    problems = []
    for row in rows:
        key = tuple(row.get(field_name) for field_name in SCENARIO)
        first = seen.setdefault(key, {})
        for name in GROUND_TRUTH:
            if name not in row:
                continue
            value, arm = first.setdefault(name, (row[name], row.get("arm")))
            if row[name] != value:
                where = " ".join(f"{f}={v}" for f, v in zip(SCENARIO, key, strict=True) if v is not None)
                problems.append(f"scenario {where}: {name}={row[name]!r} for arm {row.get('arm')}, "
                                f"{value!r} for arm {arm}")
    return problems


def batch_ids(path: Path) -> list[str]:
    try:
        return list(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return []


# The columns that identify an episode (a unit field) and the ones that legitimately differ between two runs of it.
EPISODE = ("arm", "audit", "intervention", "topology", "attack", "agents", "seed", "sink", "backbone", "task",
           "conversation", "condition", "preset", "cost_condition")
VOLATILE_PREFIXES = ("latency", "wall", "elapsed", "duration")


def episode_key(row: Mapping) -> tuple:
    return tuple(str(row.get(name)) for name in EPISODE)


def same_value(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def compare_runs(resumed: Iterable[Mapping], straight: Iterable[Mapping]) -> list[str]:
    """Rows of a run that was killed and resumed against the rows of an uninterrupted run of the same units, by
    episode. Both must hold the same episodes once, and every column but the timings must agree."""
    def index(rows: Iterable[Mapping]) -> dict[tuple, Mapping]:
        found: dict[tuple, Mapping] = {}
        seen: dict[tuple, int] = {}
        for row in rows:          # rows of one episode without a conversation field are told apart by their order
            key = episode_key(row)
            seen[key] = seen.get(key, 0) + 1
            found[(*key, seen[key])] = row
        return found

    a, b = index(resumed), index(straight)
    problems = [f"episode {k} is only in the {'resumed' if k in a else 'straight'} run" for k in sorted(a.keys() ^ b.keys())]
    for key in sorted(a.keys() & b.keys()):
        differing = [c for c in sorted(set(a[key]) | set(b[key])) if not c.startswith(VOLATILE_PREFIXES)
                     and not same_value(a[key].get(c), b[key].get(c))]
        if differing:
            problems.append(f"episode {key}: {', '.join(differing)} differ (resumed {a[key].get(differing[0])!r}, "
                            f"straight {b[key].get(differing[0])!r})")
    return problems


def all_rows(units_dir: Path) -> list[dict]:
    """The rows of every committed rows file of a block (`units_dir` is its units/ folder)."""
    return [row for path in committed_files(units_dir) for row in read_rows(path)[0]]


def tmp_files(block_dir: Path) -> set[str]:
    """The temporary files (`*.tmp`) in a block folder, as paths relative to it."""
    return {p.relative_to(block_dir).as_posix() for p in block_dir.rglob("*.tmp")} if block_dir.is_dir() else set()


def check_block(block_dir: Path, expected_units: int, rows_per_unit: int | None,
                stale_tmp: Collection[str] = ()) -> list[str]:
    """Files of a finished block: every unit done, the right number of rows, nothing torn or left over. A unit has its
    own rows file, or is one of the units a batch rows file commits. Units and rows are counted in the committed rows
    files; a rows file without its done marker is reported.

    `stale_tmp`: the temporary files that were on disk right after the kill (see `tmp_files`). CheckpointStore writes
    `<name>.tmp` and renames it, nothing reads a `.tmp`, and the store does not clean one up, so a kill between write
    and rename leaves it; the resumed launch regroups its batches and never rewrites that name. Such a file is a crash
    artifact and does not fail the check; a `.tmp` that appeared after the kill does."""
    units = block_dir / "units"
    files = committed_files(units)
    every = sorted(units.rglob("*.jsonl")) if units.is_dir() else []
    done = finished(units)
    committed = set(files)
    problems = [f"{p.relative_to(units).as_posix()}: not marked done" for p in every if p not in committed]
    found = 0
    for path in files:
        rows, torn = read_rows(path)
        batch = path.stem.startswith(BATCH)
        count = len(batch_ids(path.with_suffix(".done"))) if batch else 1
        found += count
        if torn:
            problems.append(f"{path.name}: {torn} line(s) that are not JSON")
        if rows_per_unit is None and not rows:
            problems.append(f"{path.name}: no rows")
        elif rows_per_unit is not None and len(rows) != rows_per_unit * count:
            problems.append(f"{path.name}: {len(rows)} rows, expected {rows_per_unit * count}")
    if found != expected_units:
        problems.append(f"{found} unit files, expected {expected_units}")
    problems += [f"{stem}.done has no rows" for stem in sorted(done - {p.stem for p in every})]
    # a temporary file anywhere in the block, and resumable state of a unit (state/ and state/<shard>/), each once
    leftovers = {*block_dir.rglob("*.tmp"), *(block_dir / "state").rglob("*.pt")}
    problems += [f"left over: {p.relative_to(block_dir).as_posix()}" for p in sorted(leftovers)
                 if p.relative_to(block_dir).as_posix() not in stale_tmp]
    return problems


def check_resume(before: Mapping[str, bytes], after: Mapping[str, bytes], done_before: set[str]) -> list[str]:
    """What a restart may do to the files it finds: nothing to a finished one, only append to an unfinished unit file.
    A batch rows file without its done list (an orphan: the kill came between the rows file and its done list) is not
    committed: the restart removes it before it writes (CheckpointStore.append_batch) and runs its units again, which
    check_block and compare_runs see."""
    problems = []
    for name, data in before.items():
        stem = Path(name).stem
        if stem.startswith(BATCH) and stem not in done_before:
            continue
        if name not in after:
            problems.append(f"{name}: lost after the restart")
        elif Path(name).stem in done_before:
            if after[name] != data:
                problems.append(f"{name}: a finished unit file changed in the restart")
        else:
            kept = data[: data.rfind(b"\n") + 1]
            if not after[name].startswith(kept):
                problems.append(f"{name}: rows saved before the kill were rewritten (the restart is not a prefix)")
    return problems


def check_thresholds(path: Path, expected_cells: int, alpha: float) -> list[str]:
    if not path.is_file():
        return [f"{path.name} is missing"]
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    problems = []
    if data.get("alpha") != alpha:
        problems.append(f"alpha is {data.get('alpha')!r}, settings say {alpha!r}")
    if len(cells) != expected_cells:
        problems.append(f"{len(cells)} cells, expected {expected_cells}")
    keys = [tuple(c.get(k) for k in ("arm", "topology", "agents", "backbone", "task")) for c in cells]
    problems += [f"cell {k} appears twice" for k in sorted(set(keys), key=str) if keys.count(k) > 1]
    for cell, key in zip(cells, keys, strict=True):
        found = cell.get("thresholds") or {}
        if not found:
            problems.append(f"cell {key} has no threshold")
        problems += [f"cell {key}: {name}={value!r} is not a finite number" for name, value in found.items()
                     if not (is_number(value) and math.isfinite(value))]
    return problems


def check_manifests(block_dir: Path, block: str, launches: int, units: int,
                    runtime: Mapping[str, object] | None = None) -> list[str]:
    """One manifest per launch, each recording the block, its unit count and the run-time values the plan set."""
    expected = RUNTIME_TEST_VALUES if runtime is None else runtime
    paths = sorted(block_dir.glob("manifest-*.json"))
    problems = [] if len(paths) == launches else [f"{len(paths)} manifests, expected {launches} (one per launch)"]
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("block") != block or manifest.get("units") != units:
            problems.append(f"{path.name}: block/units {manifest.get('block')}/{manifest.get('units')}, "
                            f"expected {block}/{units}")
        problems += [f"{path.name}: pre_registered does not record {key}"
                     for key in ("ni_margin_accuracy", "rank_agreement_threshold")
                     if manifest.get("pre_registered", {}).get(key) != expected[key]]
        recorded = manifest.get("settings", {}).get("runtime", {})
        if any(recorded.get(k) != v for k, v in expected.items()):
            problems.append(f"{path.name}: settings do not record the run-time test values")
        if not (isinstance(manifest.get("threads"), int) and manifest["threads"] >= 1):
            problems.append(f"{path.name}: threads not recorded")
    return problems


# ------------------------------------------------------------------------------------------ launching
@dataclass
class Run:
    code: int
    output: str


@dataclass(frozen=True)
class Interruption:
    """How the kill part-way went."""
    seen: Progress             # on disk after the last kill
    killed: bool               # False: the block finished before it could be killed
    launches: int              # launches started (each one leaves a manifest)
    cut: tuple[str, ...] = ()  # unit files the last kill left part-written (platforms with several rows per unit)


def cuts_units(rows_per_unit: int | None) -> bool:
    """Whether a kill can cut a unit in the middle: a platform with several rows per unit (LLM, one row per conversation)
    writes them one by one; the mock platform has one row per unit and commits whole batches."""
    return rows_per_unit is not None and rows_per_unit > 1


def is_guard_refusal(run: Run) -> bool:
    """The kickoff guard of swarmlab ends the launch with 'Refusing: before kickoff' or 'requires --allow-live-llm'."""
    return run.code != 0 and "Refusing:" in run.output and ("before kickoff" in run.output or "--allow-live-llm" in run.output)


@dataclass
class Session:
    plan: Plan
    launcher: Path
    results_root: Path
    profile: str
    env: dict[str, str]
    timeout: int
    tmp: Path

    def argv(self, tail: Sequence[str]) -> list[str]:
        return [sys.executable, str(self.launcher), "--results-root", str(self.results_root), *set_args(self.plan.sets),
                *tail]

    def command(self, name: str, tail: Sequence[str] = ()) -> Run:
        """`run_experiments.py <name> ...`: the subcommands take their options after their own name."""
        argv = [sys.executable, str(self.launcher), name, "--results-root", str(self.results_root),
                *set_args(self.plan.sets), *tail]
        try:
            done = subprocess.run(argv, cwd=ROOT, env=self.env, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            return Run(124, f"no result within {self.timeout} s")
        return Run(done.returncode, done.stdout + done.stderr)

    def block_dir(self, block: str) -> Path:
        return self.results_root / self.profile / block

    def launch(self, tail: Sequence[str]) -> Run:
        try:
            done = subprocess.run(self.argv(tail), cwd=ROOT, env=self.env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            return Run(124, f"no result within {self.timeout} s")
        return Run(done.returncode, done.stdout + done.stderr)

    def run_block(self, block: str) -> Run:
        run = self.launch(["--run", "--block", block, *self.plan.run_flags])
        if is_guard_refusal(run):
            raise GuardRefused(run.output.strip().splitlines()[-1])
        if run.code != 0:
            raise Abort(f"{block}: the launcher exited {run.code}\n{tail_of(run.output)}")
        return run

    def interrupt(self, evaluation: Evaluation) -> Interruption:
        """Launch the block and kill it part-way. On a platform with several rows per unit the kill must leave a unit
        part-written (some of its rows on disk, no done marker), the state a resume has to continue; a kill that leaves
        none is followed by another launch and kill, at most KILL_ATTEMPTS launches in all. A launch that finishes the block
        without a kill wipes the block folder and starts over (it counts as an attempt)."""
        directory = self.block_dir(evaluation.block)
        units_dir = directory / "units"
        per_unit = cuts_units(evaluation.rows_per_unit)
        launches = attempts = 0
        while True:
            seen, killed = self.run_and_kill(evaluation.block, evaluation.units, evaluation.rows_per_unit)
            launches += 1
            attempts += 1
            if not killed and seen.done >= evaluation.units and attempts < KILL_ATTEMPTS:
                # Under load the units left after an earlier attempt finish inside the grace period: nothing remains to
                # kill or cut. Start over on a fresh block folder (the launches counted are those since the wipe).
                shutil.rmtree(directory, ignore_errors=True)
                launches = 0
                continue
            cut = tuple(part_written(units_dir, evaluation.rows_per_unit)) if per_unit and killed else ()
            if not (per_unit and killed) or cut or attempts >= KILL_ATTEMPTS:
                return Interruption(seen, killed, launches, cut)

    def run_and_kill(self, block: str, units: int, rows_per_unit: int | None) -> tuple[Progress, bool]:
        """Launch the block and kill the launcher and its workers once enough of it is on disk. Returns
        (progress, killed).

        One row per unit (mock, whole batches committed): the kill comes once a quarter of the rows are on disk,
        preferably while a batch rows file waits for its done list. Several rows per unit (LLM): the kill comes while a
        unit file holds some but not all of its rows. A kill lands 0.1 to 0.3 s after it is asked for on Windows
        (taskkill), longer than a unit of the fake API, so the launcher is frozen first: the files are read again while
        it cannot write, and it is killed only if a unit is still part-written, else let go and watched again."""
        units_dir = self.block_dir(block) / "units"
        per_unit = cuts_units(rows_per_unit)
        threshold = 1 if per_unit else max(1, units // 4)
        baseline = progress(units_dir).rows          # a relaunch finds the rows of the launches before it
        log = self.tmp / f"{block}-kill.log"
        killed = False
        with log.open("w", encoding="utf-8") as sink:
            # A session of its own: on POSIX the launcher and its spawn workers form one process group to kill.
            proc = subprocess.Popen(self.argv(["--run", "--block", block, *self.plan.run_flags]), cwd=ROOT,
                                    env=self.env, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + self.timeout
                enough_since = 0.0
                while proc.poll() is None and time.monotonic() < deadline:
                    seen_now = progress(units_dir)
                    enough = (seen_now.rows - baseline if per_unit else seen_now.rows) >= threshold
                    if enough:
                        enough_since = enough_since or time.monotonic()
                        overdue = time.monotonic() - enough_since > IN_FLIGHT_GRACE
                        if per_unit:
                            killed = (bool(part_written(units_dir, rows_per_unit)) or overdue) and cut_unit(
                                proc, units_dir, rows_per_unit, anyway=overdue)
                        # A batch platform never shows a part-written unit; an orphan batch (rows on disk, no done
                        # list) is its in-flight commit. Do not wait for one for long.
                        elif seen_now.rows > seen_now.done or overdue:
                            kill_tree(proc)
                            killed = True
                        if killed:
                            break
                    time.sleep(0.005)
                timed_out = proc.poll() is None and not killed
                if timed_out:
                    kill_tree(proc)
                proc.wait()
            except BaseException:
                if proc.poll() is None:      # in its own session, the launcher would outlive an interrupted harness
                    kill_tree(proc)
                    proc.wait()
                raise
        seen = progress(units_dir)
        output = Run(proc.returncode, log.read_text(encoding="utf-8", errors="replace"))
        if killed:
            return seen, True
        if timed_out:
            raise Abort(f"{block}: nothing on disk after {self.timeout} s\n{tail_of(output.output)}")
        if is_guard_refusal(output):
            raise GuardRefused(output.output.strip().splitlines()[-1])
        if output.code != 0:
            raise Abort(f"{block}: the launcher exited {output.code} before it could be killed\n{tail_of(output.output)}")
        return seen, False


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the launcher and the worker processes it started. Windows: taskkill /T (a plain kill leaves the workers
    running). POSIX: SIGKILL to its process group (it runs in a session of its own), so spawn workers stop with it
    instead of committing rows after the kill."""
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        with contextlib.suppress(OSError):                  # ProcessLookupError: the group is gone already
            os.killpg(proc.pid, signal.SIGKILL)
    proc.kill()


def freeze(proc: subprocess.Popen) -> bool:
    """Stop the launcher and every process below it at once; they do not run again until `thaw` (or die with a kill).
    POSIX: SIGSTOP to its process group. Windows: psutil's suspend (NtSuspendProcess) on the launcher and its
    descendants: the venv's python.exe is a trampoline that runs the interpreter as a child process. False when it
    could not be done (the caller then kills at once and checks what the kill left)."""
    return _windows_tree(proc.pid, "suspend") if WINDOWS else _signal_group(proc, "SIGSTOP")


def thaw(proc: subprocess.Popen) -> bool:
    """Let a frozen launcher and the processes below it run again."""
    return _windows_tree(proc.pid, "resume") if WINDOWS else _signal_group(proc, "SIGCONT")


def _signal_group(proc: subprocess.Popen, name: str) -> bool:
    try:
        os.killpg(proc.pid, getattr(signal, name))
    except OSError:
        return False
    return True


def _windows_tree(pid: int, method: str) -> bool:
    """psutil's `method` (suspend or resume) on a process and its descendants (psutil's children() checks creation
    times, so a reused process id is never taken for a child). psutil 7.2.2 is a direct pin in
    pyproject.toml (also a dependency of wandb); without it the freeze is reported as not done."""
    try:
        import psutil
    except ImportError:
        return False
    try:
        root = psutil.Process(pid)
        tree = [root, *root.children(recursive=True)]
    except psutil.Error:
        return False
    done = True
    for process in tree:
        try:
            getattr(process, method)()
        except psutil.NoSuchProcess:
            continue                                     # gone already: nothing left to stop or start
        except psutil.Error:
            done = False
    return done


def cut_unit(proc: subprocess.Popen, units_dir: Path, rows_per_unit: int, anyway: bool) -> bool:
    """Freeze the launcher and kill it if a unit is still part-written (or `anyway`); else let it run on. Returns whether
    it was killed. Where it cannot be frozen it is killed at once and the caller checks what the kill left."""
    held = freeze(proc)
    if held:
        time.sleep(FREEZE_SETTLE)
    if not held or anyway or part_written(units_dir, rows_per_unit) or not thaw(proc):
        kill_tree(proc)
        return True
    return False


def tail_of(text: str, lines: int = 12) -> str:
    return "\n".join("        " + line for line in text.strip().splitlines()[-lines:])


# ------------------------------------------------------------------------------------------------ stages
def stage_list_and_dry_run(session: Session, report: Report) -> None:
    blocks = [c.block for c in session.plan.calibrations] + [e.block for e in session.plan.evaluations]
    report.section("list blocks")
    run = session.launch(["--list"])
    report.check(run.code == 0, f"--list exits 0 (got {run.code})")
    missing = [b for b in blocks if b not in run.output]
    report.check(not missing, "--list names every block of the run" + (f" (missing {missing})" if missing else ""))
    report.section("dry run")
    run = session.launch(["--dry-run", *[a for b in blocks for a in ("--block", b)]])
    report.check(run.code == 0, f"--dry-run exits 0 (got {run.code})")
    for evaluation in session.plan.evaluations:
        line = next((ln for ln in run.output.splitlines() if ln.startswith(evaluation.block + " ")), "")
        counted = line.split()[1].replace(",", "") if len(line.split()) > 1 else "?"
        report.check(counted == str(evaluation.units),
                     f"--dry-run counts {counted} units for {evaluation.block}, the grid has {evaluation.units}")


def stage_calibration(session: Session, report: Report, calibration: Calibration, alpha: float) -> None:
    report.section(calibration.block)
    session.run_block(calibration.block)
    directory = session.block_dir(calibration.block)
    report.problems("thresholds.json", check_thresholds(directory / "thresholds.json", calibration.cells, alpha))
    report.problems("manifest", check_manifests(directory, calibration.block, 1, calibration.units,
                                                  plan_runtime(session.plan.sets)))


def straight_session(session: Session) -> Session:
    """A second results root holding only the calibration of the first, to run the same units without a kill."""
    root = session.tmp / "straight"
    if not root.exists():
        for calibration in session.plan.calibrations:
            shutil.copytree(session.block_dir(calibration.block),
                            root / "results" / session.profile / calibration.block)
    return dataclasses.replace(session, results_root=root / "results")


def stage_evaluation(session: Session, report: Report, evaluation: Evaluation) -> None:
    block, directory = evaluation.block, session.block_dir(evaluation.block)
    units_dir = directory / "units"
    launches = 1
    stale_tmp: set[str] = set()
    if evaluation.interrupt:
        report.section(f"{block}: killed part-way")
        kill = session.interrupt(evaluation)
        again = f" (launch {kill.launches} of at most {KILL_ATTEMPTS})" if kill.launches > 1 else ""
        if not report.check(kill.killed and kill.seen.done < evaluation.units,
                            f"killed with {kill.seen.done} of {evaluation.units} units done ({kill.seen.started} "
                            f"started){again}"):
            raise Abort(f"{block} finished before it could be killed{again}; enlarge the grid (--seeds, --agents)")
        if cuts_units(evaluation.rows_per_unit):
            if kill.cut:
                report.ok(f"the kill left {len(kill.cut)} unit(s) part-written ({', '.join(kill.cut)}): the resume "
                          "continues a unit after its last saved row")
            else:
                report.fail(f"resume path not exercised: none of {kill.launches} kill(s) left a unit part-written "
                            f"(some of its {evaluation.rows_per_unit} rows on disk, no done marker)")
        before, done_before = snapshot(units_dir), finished(units_dir)
        stale_tmp = tmp_files(directory)             # crash artifacts of the kill, tolerated by check_block below
        if stale_tmp:
            report.ok(f"the kill left {len(stale_tmp)} temporary file(s) ({', '.join(sorted(stale_tmp))}): a crash "
                      "artifact the store does not clean up, tolerated")
        report.section(f"{block}: resumed")
        session.run_block(block)
        report.problems("restart keeps what was saved", check_resume(before, snapshot(units_dir), done_before))
        launches = kill.launches + 1
    else:
        report.section(block)
        session.run_block(block)
    report.problems("unit files and rows", check_block(directory, evaluation.units, evaluation.rows_per_unit, stale_tmp))
    if evaluation.metrics:
        report.problems("rows are sane", check_rows(units_dir, columns()))
        report.problems("arms of one scenario share its ground truth", check_scenarios(all_rows(units_dir)))
    if evaluation.interrupt:
        straight = straight_session(session)
        report.section(f"{block}: the same units without a kill")
        straight.run_block(block)
        report.problems("the resumed run equals the uninterrupted run, episode by episode",
                        compare_runs(all_rows(units_dir), all_rows(straight.block_dir(block) / "units")))
    report.section(f"{block}: second launch changes nothing")
    before = snapshot(units_dir)
    session.run_block(block)
    report.check(snapshot(units_dir) == before and finished(units_dir) == {p.stem for p in units_dir.rglob("*.jsonl")},
                 "no unit file changed and none was added")
    report.problems("manifests", check_manifests(directory, block, launches + 1, evaluation.units,
                                              plan_runtime(session.plan.sets)))


def has_run_all(session: Session) -> bool:
    """Whether the launcher offers the `status` and `run-all` commands."""
    out = session.launch(["--help"]).output
    return "run-all" in out and "status" in out


def stage_status_and_dry_run_all(session: Session, report: Report) -> bool:
    report.section("status and run-all --dry-run")
    if not has_run_all(session):
        print("  SKIP  this launcher has no status / run-all command (known gap, see scripts/e2e_common.py)", flush=True)
        return False
    blocks = [c.block for c in session.plan.calibrations] + [e.block for e in session.plan.evaluations]
    run = session.command("status")
    report.check(run.code == 0, f"status exits 0 (got {run.code})")
    missing = [b for b in blocks if b not in run.output]
    report.check(not missing, "status names every block of the run" + (f" (missing {missing})" if missing else ""))
    run = session.command("run-all", ["--dry-run", *session.plan.run_flags])
    report.check(run.code == 0, f"run-all --dry-run exits 0 (got {run.code})")
    return True


def stage_run_all_after_everything(session: Session, report: Report) -> None:
    """Every block is finished: run-all must change nothing, and status must still work."""
    report.section("run-all on a finished run")
    before = {e.block: snapshot(session.block_dir(e.block) / "units") for e in session.plan.evaluations}
    run = session.command("run-all", session.plan.run_flags)
    report.check(run.code == 0, f"run-all exits 0 (got {run.code})")
    if run.code:
        print(tail_of(run.output))
    after = {e.block: snapshot(session.block_dir(e.block) / "units") for e in session.plan.evaluations}
    report.check(before == after, "run-all on a finished run changes no unit file")
    report.check(session.command("status").code == 0, "status exits 0 after the run")


def stage_analysis(session: Session, report: Report, entry: str) -> object:
    report.section("analysis")
    module_name, _, function = entry.partition(":")
    try:
        function_ = getattr(importlib.import_module(module_name), function)
    except ImportError as error:
        raise Abort(f"analysis: {module_name} cannot be imported ({error}); the analysis feature is not built") from error
    except AttributeError as error:
        raise Abort(f"analysis: {module_name} does not export {function}(results_dir, settings)") from error
    from bazt_lab.core import public as core
    overrides = override_items({**session.plan.sets, **session.plan.analysis_sets})
    settings = core.load_settings(profile=session.profile, overrides=overrides)
    try:
        result = function_(session.results_root / session.profile, settings)
    except core.Unset as missing:
        raise Abort(f"analysis: setting not set: {missing.args[0]}") from missing
    try:
        text = json.dumps(result, default=str)
    except TypeError as error:
        raise Abort(f"analysis: the result cannot be written as JSON ({error})") from error
    if report.check(result not in (None, {}, [], ()) and len(text) > 2,
                    f"analysis returned a result ({len(text)} characters)"):
        ran = [e.block for e in session.plan.evaluations if e.metrics]
        report.problems("the analysis read every committed row of the blocks it reads",
                        check_analysis(result, session.results_root / session.profile, ran))
    return result


def check_analysis(result: object, results_dir: Path, ran: Sequence[str]) -> list[str]:
    """What the analysis says it read (meta.platforms of its report) against the files on disk. For each platform, the
    rows it read must be the rows of the committed rows files of the blocks it reads for that platform. Every metrics
    block the run wrote (`ran`) must be read by a platform, must not be in its blocks_missing, and no hypothesis may
    report no_data for it. The real analysis returns a full report even when it reads nothing, so a non-empty result
    proves nothing. A run-all may have written blocks the plan does not name (the mock blocks of the LLM run): they
    count with the platform that reads them."""
    meta = result.get("meta") if isinstance(result, Mapping) else None
    platforms = meta.get("platforms") if isinstance(meta, Mapping) else None
    if not isinstance(platforms, Mapping):
        return ["the result has no meta.platforms: it does not say which rows it read"]
    info = {name: entry if isinstance(entry, Mapping) else {} for name, entry in platforms.items()}
    problems = []
    for name, entry in info.items():
        blocks = list(entry.get("blocks", []))
        on_disk = sum(committed_rows(results_dir / block) for block in blocks)
        if entry.get("rows") != on_disk:
            skipped = entry.get("uncommitted_files")
            problems.append(f"{name}: the analysis read {entry.get('rows')} rows, the committed rows files of "
                            f"{', '.join(blocks) or 'no block'} hold {on_disk}"
                            + (f" ({skipped} uncommitted rows file(s) not read)" if skipped else ""))
    for block in ran:
        readers = [name for name, entry in info.items() if block in entry.get("blocks", [])]
        if not readers:
            problems.append(f"{block}: the run wrote it but no platform of the analysis reads it")
        problems += [f"{block}: in {name}.blocks_missing, the analysis read no row of it" for name in readers
                     if block in info[name].get("blocks_missing", [])]
    hypotheses = result.get("hypotheses") if isinstance(result, Mapping) else None
    for hypothesis, by_platform in (hypotheses.items() if isinstance(hypotheses, Mapping) else ()):
        for name, entry in (by_platform.items() if isinstance(by_platform, Mapping) else ()):
            lost = sorted(set(entry.get("blocks_missing", [])) & set(ran)) if isinstance(entry, Mapping) and entry.get(
                "status") == "no_data" else []
            if lost:
                problems.append(f"{hypothesis} {name}: no_data, though the run wrote {', '.join(lost)}")
    return problems


def active_profile(profile: str | None) -> tuple[str, float]:
    data = tomllib.loads(PROFILES.read_text(encoding="utf-8"))
    return profile or data["active_profile"], data["budgets"]["alpha"]


def isolated_paths(tmp: Path) -> dict[str, object]:
    """The settings that point at folders inside the repository (results/experiments/gnn/...), moved into the temporary
    directory so a smoke run neither leaves files in the repository nor trips over those of an earlier run."""
    return {"gnn_train.logs_dir": str(tmp / "gnn" / "logs"), "gnn.artifacts_dir": str(tmp / "gnn" / "checkpoints")}


def execute(plan: Plan, args: argparse.Namespace) -> int:
    """Run the plan; 0 passed, 1 a check failed, 2 the run could not start, 3 the kickoff guard refused."""
    print(f"e2e {plan.name}: launcher {args.launcher}")
    if not Path(args.launcher).is_file():
        print(f"launcher {args.launcher} does not exist")
        return 2
    text = PROFILES.read_text(encoding="utf-8")
    missing = missing_test_values(text) + missing_arm_values(text)
    if missing:
        print(f"profiles.toml documents run-time variables with no E2E test value: {', '.join(missing)}; add them to "
              "RUNTIME_TEST_VALUES or ARM_TEST_VALUES in scripts/e2e_common.py")
        return 2
    profile, alpha = active_profile(args.profile)
    tmp = Path(tempfile.mkdtemp(prefix=f"bazt_e2e_{plan.name}_"))
    plan.sets.update(isolated_paths(tmp))
    session = Session(plan, Path(args.launcher), tmp / "results", profile,
                      child_env(os.environ, plan.kickoff_override), args.timeout, tmp)
    print(f"results in {session.results_root} (profile {profile}); the run-time variables use E2E TEST VALUES")
    report = Report()
    code = 1
    try:
        stage_list_and_dry_run(session, report)
        with_run_all = stage_status_and_dry_run_all(session, report)
        for calibration in plan.calibrations:
            stage_calibration(session, report, calibration, alpha)
        for evaluation in plan.evaluations:
            stage_evaluation(session, report, evaluation)
        if with_run_all:
            stage_run_all_after_everything(session, report)
        stage_analysis(session, report, args.analysis_entry)
        code = 1 if report.failures else 0
    except Abort as stop:
        report.fail(str(stop))
    except GuardRefused as refusal:
        print(f"\nThe launcher refused to start an LLM block: {refusal}")
        print("The kickoff guard blocks every LLM block until the official start, fake API or not. Run this after "
              "kickoff, or accept the test explicitly with --kickoff-override <ISO time in the past>.")
        code = 3
    finally:
        if args.keep or code != 0:
            print(f"results kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    verdict = "passed" if code == 0 else f"FAILED ({len(report.failures)} problem(s))" if code == 1 else "stopped"
    print(f"\ne2e {plan.name}: {verdict}")
    return code


def main_for(name: str, parser_factory, plan_factory, argv=None) -> int:
    """Shared `main` of the two smoke scripts."""
    args = parser_factory().parse_args(argv)
    return execute(plan_factory(args), args)
