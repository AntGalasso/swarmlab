#!/usr/bin/env python3
"""run_experiments.py -- launcher for the BAZT experiments (EXPERIMENT_PLAN.md).

Every setting comes from experiments/profiles.toml; nothing here has a default. Two profiles size the LLM grid
(spec §1.1): `sprint` and `research`.

  python run_experiments.py --list                       blocks, unit counts, missing settings and components
  python run_experiments.py --dry-run                    sizes: units, conversations, LLM calls, hours
  python run_experiments.py --run --block mock_frontier  launch one block (resumable)
  python run_experiments.py --profile research --set llm.conversations_per_cell=80 --dry-run

A block is a list of units. An LLM unit is one seed run (the episode of the spec): its conversations in sequence on
one graph. A mock unit is one episode. Each unit is handed to the component named in [components]:

  run_unit(unit: dict, settings: dict) -> list[dict]        one row per conversation (LLM) or per episode (mock)
  calibrate_unit(unit: dict, settings: dict) -> list[dict]  attack-free calibration rows
  train_unit(unit: dict, settings: dict) -> list[dict]      GNN training rows

Rows are written to results/experiments/<profile>/<block>/units/<unit id>.jsonl; a unit whose file exists is skipped,
so an interrupted block resumes where it stopped.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import itertools
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Iterator, List

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "experiments" / "profiles.toml"
RESULTS_ROOT = ROOT / "results" / "experiments"
FREEZE_MARKER = RESULTS_ROOT / "ANALYSIS_FROZEN"   # created by hand when the analysis is frozen

# Run-time variables (spec §10) each platform needs before a --run. The two pre-registered ones are required for
# every run, so they are recorded in the manifest before any result exists.
PRE_REGISTERED = ("ni_margin_accuracy", "rank_agreement_threshold")
RUNTIME_BY_PLATFORM = {
    "llm": ("n_insiders", "rate_limit_per_identity", "recovery_margin", "random_edge_probability", "ba_m", "ws_k",
            "ws_rewire_p", "llm_seeds", "llm_calibration_seeds"),
    "mock": ("n_insiders", "rate_limit_per_identity", "horizon_constant", "eps_correct", "recovery_margin", "ba_m",
             "ws_k", "ws_rewire_p"),
}


class Unset(KeyError):
    """A setting the block needs is not set in profiles.toml."""


# ================================================================================== settings
def _put(tree: dict, dotted: str, value) -> None:
    *path, leaf = dotted.split(".")
    for part in path:
        tree = tree.setdefault(part, {})
    tree[leaf] = value


def _parse_value(text: str):
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def load_settings(path: Path = SETTINGS_PATH, profile: str | None = None, overrides=()) -> dict:
    """profiles.toml, with the chosen profile applied on top, then each `key=value` override."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    name = profile or raw.get("active_profile")
    profiles = raw.get("profiles", {})
    if name not in profiles:
        sys.exit(f"Unknown profile {name!r}; profiles.toml defines {sorted(profiles)}.")
    settings = {k: v for k, v in raw.items() if k not in ("profiles", "active_profile")}
    for key, value in profiles[name].items():
        _put(settings, key, value)
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep:
            sys.exit(f"--set expects key=value, got {item!r}")
        _put(settings, key.strip(), _parse_value(text.strip()))
    settings["profile"] = name
    return settings


def get(settings: dict, dotted: str):
    node = settings
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise Unset(dotted)
        node = node[part]
    return node


# ===================================================================================== units
def unit_id(unit: dict) -> str:
    digest = hashlib.sha256(json.dumps(unit, sort_keys=True).encode()).hexdigest()[:12]
    return f"{unit['arm']}-{digest}"


def _cell(value) -> float | str:
    """A budget level: a percent from profiles.toml becomes a fraction of deliveries; "uncapped" stays."""
    return value if value == "uncapped" else float(value) / 100.0


def _capped_cells(levels) -> List[tuple]:
    """Every (audit, intervention) pair of the crossed levels except the fully uncapped one."""
    axis = [_cell(v) for v in levels] + ["uncapped"]
    return [(a, i) for a, i in itertools.product(axis, axis) if not (a == "uncapped" and i == "uncapped")]


def _seed_range(bounds) -> List[int]:
    lo, hi = bounds
    return list(range(int(lo), int(hi) + 1))


def _llm_seeds(s: dict, key: str, strict: bool) -> List:
    try:
        return list(get(s, f"runtime.{key}"))
    except Unset:
        if strict:
            raise
        return [f"<{key}[{i}]>" for i in range(get(s, "llm.seeds_per_cell"))]


def _backbones(s: dict, strict: bool) -> List[str]:
    try:
        return list(get(s, "llm.backbones"))
    except Unset:
        if strict:
            raise
        return [f"<backbone {i + 1}>" for i in range(get(s, "llm.n_backbones"))]


def _per_seed(s: dict, total_key: str) -> int:
    total, seeds = get(s, total_key), get(s, "llm.seeds_per_cell")
    if total % seeds:
        sys.exit(f"{total_key}={total} is not divisible by llm.seeds_per_cell={seeds}.")
    return total // seeds


def _llm_attack_cells(s: dict) -> List[tuple]:
    attacks = [tuple(c) for c in get(s, "llm.attack_cells")]
    return attacks + [("none", task) for task in get(s, "llm.attack_free_tasks")]


def _llm_units(s: dict, block: str, arms, cells, strict: bool) -> List[dict]:
    conv = _per_seed(s, "llm.conversations_per_cell")
    units = []
    for arm, (audit, interv), backbone, topology, (attack, task), seed in itertools.product(
            arms, cells, _backbones(s, strict), get(s, "llm.topologies"), _llm_attack_cells(s),
            _llm_seeds(s, "llm_seeds", strict)):
        units.append(dict(block=block, platform="llm", arm=arm, audit=audit, intervention=interv, backbone=backbone,
                          topology=topology, attack=attack, task=task, seed=seed, conversations=conv,
                          agents=get(s, "llm.agents"), rounds=get(s, "llm.rounds")))
    return units


def units_llm_headline(s, strict):
    return _llm_units(s, "llm_headline", get(s, "llm.headline_arms"), [("uncapped", "uncapped")], strict)


def units_llm_capped(s, strict):
    cells = [(_cell(a), _cell(i)) for a, i in get(s, "llm.capped_cells")]
    return _llm_units(s, "llm_capped", get(s, "llm.capped_arms"), cells, strict)


def units_llm_calibration(s, strict):
    conv = _per_seed(s, "llm.calibration_conversations_per_cell")
    return [dict(block="llm_calibration", platform="llm", arm=arm, audit="uncapped", intervention="uncapped",
                 backbone=b, topology=t, attack="none", task=task, seed=seed, conversations=conv,
                 agents=get(s, "llm.agents"), rounds=get(s, "llm.rounds"))
            for arm, b, t, task, seed in itertools.product(
                get(s, "llm.calibrated_arms"), _backbones(s, strict), get(s, "llm.topologies"),
                get(s, "llm.attack_free_tasks"), _llm_seeds(s, "llm_calibration_seeds", strict))]


def units_gnn_train(s, strict):
    try:
        backbone = get(s, "gnn_train.backbone")
    except Unset:
        if strict:
            raise
        backbone = "<gnn backbone>"
    return [dict(block="gnn_train", platform="llm", arm="GNN", attack=attack, task=task, edge_probability=p,
                 attackers=k, backbone=backbone, conversations=get(s, "gnn_train.conversations_per_setting"),
                 agents=get(s, "gnn_train.agents"), rounds=get(s, "gnn_train.rounds"))
            for (attack, task), p, k in itertools.product(
                [tuple(f) for f in get(s, "gnn_train.families")], get(s, "gnn_train.edge_probabilities"),
                get(s, "gnn_train.attackers"))]


def units_llm_adaptive_check(s, strict):
    arms, families, condition = (get(s, f"llm.adaptive_check.{k}") for k in ("arms", "families", "condition"))
    return [dict(block="llm_adaptive_check", platform="llm", arm=arm, audit="uncapped", intervention="uncapped",
                 backbone=b, topology=t, attack=fam, task="mmlu", condition=condition, seed=seed,
                 conversations=_per_seed(s, "llm.conversations_per_cell"), agents=get(s, "llm.agents"),
                 rounds=get(s, "llm.rounds"))
            for arm, fam, b, t, seed in itertools.product(arms, families, _backbones(s, strict),
                                                          get(s, "llm.topologies"),
                                                          _llm_seeds(s, "llm_seeds", strict))]


def _mock_units(s, block, arms_cells, topologies, families, agent_counts, seeds, extra=None) -> Iterator[dict]:
    """arms_cells: list of (arm, (audit, intervention)). Every agent takes the sink role when mock.every_sink.

    A generator: the frontier alone has millions of episodes.
    """
    every_sink = get(s, "mock.every_sink")
    for (arm, (audit, interv)), topo, fam, n, seed in itertools.product(arms_cells, topologies, families,
                                                                        agent_counts, seeds):
        for sink in (range(n) if every_sink else [None]):
            yield dict(block=block, platform="mock", arm=arm, audit=audit, intervention=interv, topology=topo,
                       attack=fam, agents=n, seed=seed, sink=sink, **(extra or {}))


def _frontier_arms_cells(s) -> List[tuple]:
    capped = _capped_cells(get(s, "budgets.levels_percent"))
    pairs = [(arm, ("uncapped", "uncapped")) for arm in get(s, "mock.uncapped_arms")]
    pairs += [(arm, c) for arm in get(s, "mock.capped_arms") + get(s, "mock.allocation_arms") for c in capped]
    return pairs


def units_mock_frontier(s, strict, seeds_key="mock.main_seeds", block="mock_frontier"):
    return _mock_units(s, block, _frontier_arms_cells(s), get(s, "mock.topologies"),
                       get(s, "mock.families") + ["none"], get(s, "mock.frontier_agents"),
                       _seed_range(get(s, seeds_key)))


def units_mock_heldout(s, strict):
    return units_mock_frontier(s, strict, seeds_key="mock.heldout_seeds", block="mock_heldout")


def units_mock_calibration(s, strict):
    return [dict(block="mock_calibration", platform="mock", arm=arm, audit="uncapped", intervention="uncapped",
                 topology=t, attack="none", agents=n, seed=seed, sink=None)
            for arm, t, n, seed in itertools.product(get(s, "mock.calibrated_arms"), get(s, "mock.topologies"),
                                                     get(s, "mock.epidemiology_agents"),
                                                     _seed_range(get(s, "mock.tune_seeds")))]


def _scoped(name: str, topologies_key="mock.topologies", agents_key="mock.frontier_agents",
            variant_key=None, variant_field=None):
    """A mock block whose arms and cells are set in [mock.<name>]; it raises Unset until both are set."""
    def build(s, strict):
        arms_cells = [(a, (_cell(au), _cell(iv))) for a in get(s, f"mock.{name}.arms")
                      for au, iv in get(s, f"mock.{name}.cells")]
        args = (get(s, topologies_key), get(s, "mock.families") + ["none"], get(s, agents_key),
                _seed_range(get(s, "mock.main_seeds")))
        variants = get(s, variant_key) if variant_key else [None]
        return itertools.chain.from_iterable(
            _mock_units(s, f"mock_{name}", arms_cells, *args, {variant_field: v} if variant_key else None)
            for v in variants)
    return build


# ==================================================================================== blocks
@dataclass(frozen=True)
class Block:
    name: str
    platform: str
    component: str                      # key in [components]
    description: str
    units: Callable[[dict, bool], Iterable[dict]]   # strict=True for a run: placeholders are not allowed
    requires_freeze: bool = False


BLOCKS: Dict[str, Block] = {b.name: b for b in [
    Block("llm_calibration", "llm", "llm_calibration", "attack-free calibration of B, C, BAZT (alpha deny rate)",
          units_llm_calibration),
    Block("gnn_train", "llm", "gnn_train", "G-Safeguard GNN training, upstream setup", units_gnn_train),
    Block("llm_headline", "llm", "llm_unit", "uncapped A, B, C, BAZT, GNN", units_llm_headline),
    Block("llm_capped", "llm", "llm_unit", "capped C and BAZT at the profile's budget cells", units_llm_capped),
    Block("llm_adaptive_check", "llm", "llm_unit", "held-out LLM adaptive check", units_llm_adaptive_check),
    Block("mock_calibration", "mock", "mock_calibration", "attack-free calibration on the tune split",
          units_mock_calibration),
    Block("mock_frontier", "mock", "mock_unit", "full 7x7 budget frontier, every sink", units_mock_frontier),
    Block("mock_epidemiology", "mock", "mock_unit", "H-topology at 12 and 100 agents",
          _scoped("epidemiology", topologies_key="mock.epidemiology.topologies",
                  agents_key="mock.epidemiology_agents")),
    Block("mock_ablations", "mock", "mock_unit", "BAZT minus one component", _scoped("ablations")),
    Block("mock_adaptive", "mock", "mock_unit", "adaptive attacker conditions",
          _scoped("adaptive", variant_key="mock.adaptive.conditions", variant_field="condition")),
    Block("mock_presets_robustness", "mock", "mock_unit", "robustness auditor presets",
          _scoped("presets_robustness", variant_key="mock.presets_robustness.presets", variant_field="preset")),
    Block("mock_costs_robustness", "mock", "mock_unit", "CAGE-4 cost weights", _scoped("costs_robustness")),
    Block("mock_legacy", "mock", "mock_unit", "existing swarmlab arms with matched-timing controls",
          _scoped("legacy")),
    Block("mock_heldout", "mock", "mock_unit", "frontier on held-out seeds 31-60, after the freeze",
          units_mock_heldout, requires_freeze=True),
]}


# ================================================================================== sizing
def llm_calls(unit: dict, s: dict) -> dict:
    """LLM calls of one unit. Judge calls are an upper bound for capped arms (every sent text checked)."""
    texts = unit["conversations"] * unit["agents"] * unit["rounds"]
    judge = texts if unit["block"] != "gnn_train" and unit["arm"] in get(s, "llm.judge_arms") else 0
    adversary = (unit["conversations"] * unit["rounds"] * get(s, "estimates.aitm_adversary_calls_per_round")
                 if unit.get("attack") == "aitm" else 0)
    return {"agent": texts, "judge": judge, "adversary": adversary}


def cache_savings(units: List[dict], s: dict) -> int:
    """Round-0 calls shared by arms that run the same conversation (same everything except arm and budget)."""
    groups: Dict[tuple, List[dict]] = {}
    for u in units:
        if u["block"] in ("llm_headline", "llm_capped"):
            key = tuple((k, str(v)) for k, v in sorted(u.items()) if k not in ("arm", "audit", "intervention", "block"))
            groups.setdefault(key, []).append(u)
    saved = 0
    judge_arms = set(get(s, "llm.judge_arms"))
    for members in groups.values():
        first = members[0]
        round0 = first["conversations"] * first["agents"]
        saved += (len(members) - 1) * round0
        saved += max(0, sum(1 for m in members if m["arm"] in judge_arms) - 1) * round0
    return saved


def _usd(calls: dict, s: dict) -> float | None:
    prices = get(s, "estimates.usd_per_million_tokens")
    try:
        backbones = list(get(s, "llm.backbones"))
    except Unset:
        backbones = [m for m in prices if m != get(s, "llm.judge.model")]
    agent_in = sum(prices[b][0] for b in backbones) / len(backbones)
    agent_out = sum(prices[b][1] for b in backbones) / len(backbones)
    judge_in = prices[get(s, "llm.judge.model")][0]
    agent_calls = calls["agent"] + calls["adversary"]
    return (agent_calls * (get(s, "estimates.input_tokens_per_agent_call") * agent_in
                           + get(s, "estimates.output_tokens_per_agent_call") * agent_out)
            + calls["judge"] * get(s, "estimates.input_tokens_per_judge_call") * judge_in) / 1e6


def dry_run(s: dict, names: List[str]) -> None:
    rate = get(s, "estimates.calls_per_second")
    per_episode = {int(k): v for k, v in get(s, "estimates.mock_seconds_per_episode").items()}
    print(f"profile: {s['profile']}\n")
    print(f"{'block':<26}{'units':>10}{'conv/episodes':>15}{'LLM calls':>13}{'USD est':>10}{'hours':>9}")
    all_llm: List[dict] = []
    totals = {"agent": 0, "judge": 0, "adversary": 0}
    for name in names:
        block = BLOCKS[name]
        try:
            units = block.units(s, False)
            if block.platform == "llm":
                units = list(units)
        except Unset as missing:
            print(f"{name:<26}{'scope not set: ' + missing.args[0]:>57}")
            continue
        if block.platform == "llm":
            calls = {"agent": 0, "judge": 0, "adversary": 0}
            for u in units:
                for k, v in llm_calls(u, s).items():
                    calls[k] += v
            all_llm += units
            for k in totals:
                totals[k] += calls[k]
            n = sum(calls.values())
            print(f"{name:<26}{len(units):>10,}{sum(u['conversations'] for u in units):>15,}{n:>13,}"
                  f"{_usd(calls, s):>10,.0f}{n / rate / 3600:>9.1f}")
        else:
            by_n: Dict[int, int] = {}
            for u in units:
                by_n[u["agents"]] = by_n.get(u["agents"], 0) + 1
            count = sum(by_n.values())
            known = all(n in per_episode for n in by_n)
            hours = f"{sum(c * per_episode[n] for n, c in by_n.items()) / 3600:.1f}" if known else "n/a"
            print(f"{name:<26}{count:>10,}{count:>15,}{'0':>13}{'0':>10}{hours:>9}")
    total = sum(totals.values())
    if total:
        saved = cache_savings(all_llm, s)
        print(f"\nLLM calls: {total:,} (agents {totals['agent']:,}, judge {totals['judge']:,} upper bound, "
              f"adversary {totals['adversary']:,}); about {total / rate / 3600:.1f} h at {rate} calls/s")
        print(f"with the request cache (round 0 shared across arms): about {total - saved:,} calls, "
              f"{(total - saved) / rate / 3600:.1f} h")


# ===================================================================================== launch
def _component(s: dict, key: str) -> Callable:
    target = get(s, f"components.{key}")
    module, _, func = target.partition(":")
    try:
        return getattr(importlib.import_module(module), func)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"component {key} = {target!r} is not available ({e}); see spec section 8") from e


def missing_runtime(s: dict, platform: str) -> List[str]:
    keys = list(PRE_REGISTERED) + list(RUNTIME_BY_PLATFORM[platform])
    missing = [k for k in keys if _is_unset(s, f"runtime.{k}")]
    if platform == "llm":
        missing += [f"llm_sink_positions.{t}" for t in get(s, "llm.topologies")
                    if _is_unset(s, f"runtime.llm_sink_positions.{t}")]
    return missing


def _is_unset(s: dict, key: str) -> bool:
    try:
        get(s, key)
        return False
    except Unset:
        return True


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_block(s: dict, name: str, allow_live_llm: bool) -> int:
    block = BLOCKS[name]
    if block.requires_freeze and not FREEZE_MARKER.exists():
        sys.exit(f"Refusing {name}: the analysis is not frozen ({FREEZE_MARKER} does not exist).")
    missing = missing_runtime(s, block.platform)
    if missing:
        sys.exit(f"Refusing {name}: run-time variables not set in profiles.toml [runtime]: {', '.join(missing)}")
    if block.platform == "llm":
        import swarmlab  # kickoff guard: no live LLM call before the hackathon starts
        swarmlab._kickoff_guard(SimpleNamespace(mode="llm", allow_live_llm=allow_live_llm))
    try:
        units = block.units(s, True)
        fn = _component(s, block.component)
    except Unset as e:
        sys.exit(f"Refusing {name}: setting not set in profiles.toml: {e.args[0]}")
    except RuntimeError as e:
        sys.exit(f"Refusing {name}: {e}")

    out = RESULTS_ROOT / s["profile"] / name
    (out / "units").mkdir(parents=True, exist_ok=True)
    manifest = {"block": name, "profile": s["profile"], "git_commit": _git_commit(),
                "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "pre_registered": {k: get(s, f"runtime.{k}") for k in PRE_REGISTERED}, "settings": s}
    (out / f"manifest-{manifest['started_at'].replace(':', '')}.json").write_text(json.dumps(manifest, indent=2))

    done = skipped = 0
    for unit in units:
        path = out / "units" / f"{unit_id(unit)}.jsonl"
        if path.exists():
            skipped += 1
            continue
        rows = fn(unit, s)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps({"unit": unit, **row}) + "\n" for row in rows), encoding="utf-8")
        os.replace(tmp, path)
        done += 1
    print(f"{name}: {done} units run, {skipped} already done, results in {out}")
    return 0


def list_blocks(s: dict) -> None:
    print(f"profile: {s['profile']}\n")
    print(f"{'block':<26}{'platform':<10}{'units':>10}  {'component':<14}needs")
    for name, block in BLOCKS.items():
        try:
            count = f"{sum(1 for _ in block.units(s, False)):,}"
        except Unset as e:
            count = "-"
            needs = f"set {e.args[0]}"
        else:
            needs = ""
        try:
            _component(s, block.component)
            comp = "ok"
        except RuntimeError:
            comp = "not built"
        runtime = missing_runtime(s, block.platform)
        if runtime:
            needs = (needs + "; " if needs else "") + f"{len(runtime)} run-time variables"
        print(f"{name:<26}{block.platform:<10}{count:>10}  {comp:<14}{needs}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--settings", default=str(SETTINGS_PATH), help="settings file (default: experiments/profiles.toml)")
    ap.add_argument("--profile", default=None, help="sprint or research (default: active_profile in the settings)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one setting for this launch, e.g. --set llm.conversations_per_cell=80")
    ap.add_argument("--block", action="append", default=[], choices=list(BLOCKS), help="block(s) to act on")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list blocks with unit counts and what they still need")
    mode.add_argument("--dry-run", action="store_true", help="print sizes without running anything")
    mode.add_argument("--run", action="store_true", help="run the given block(s)")
    ap.add_argument("--allow-live-llm", action="store_true", help="required for LLM blocks (after kickoff only)")
    args = ap.parse_args(argv)

    s = load_settings(Path(args.settings), args.profile, args.set)
    if args.list:
        list_blocks(s)
        return 0
    names = args.block or list(BLOCKS)
    if args.dry_run:
        dry_run(s, names)
        return 0
    if not args.block:
        sys.exit("--run needs at least one --block.")
    for name in names:
        run_block(s, name, args.allow_live_llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
