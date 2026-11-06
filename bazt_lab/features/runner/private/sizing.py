"""Dry-run sizing: units, conversations or episodes, LLM calls, dollars and hours of every block, before running any."""
from __future__ import annotations

from typing import Dict, List, Mapping

from bazt_lab.core import public as core
from bazt_lab.features.runner.private.blocks import BLOCKS

CALL_KINDS = ("agent", "judge", "adversary", "rewrite", "ma_judge")


def llm_calls(unit: dict, s: Mapping) -> dict:
    """LLM calls of one unit. Judge calls are an upper bound for capped arms (every sent text checked)."""
    texts = unit["conversations"] * unit["agents"] * unit["rounds"]
    judge = texts if unit["block"] != "gnn_train" and unit["arm"] in core.get(s, "llm.judge_arms") else 0
    adversary = (unit["conversations"] * unit["rounds"] * core.get(s, "estimates.aitm_adversary_calls_per_round")
                 if unit.get("attack") == "aitm" else 0)
    rewrite = (round(texts * core.get(s, "estimates.rewrite_calls_share_of_agent_calls"))
               if unit.get("sanitize_mode") == "llm_rewrite" else 0)
    ma_judge = (unit["conversations"] * unit["agents"] * core.get(s, "estimates.ma_judge_calls_per_agent")
                if unit.get("attack") == "insider_ma" and "llm_judge" in core.get(s, "llm.ma_correctness") else 0)
    return {"agent": texts, "judge": judge, "adversary": adversary, "rewrite": rewrite, "ma_judge": ma_judge}


def cache_savings(units: List[dict], s: Mapping) -> int:
    """Round-0 calls shared by arms that run the same conversation (same everything except arm and budget)."""
    groups: Dict[tuple, List[dict]] = {}
    for unit in units:
        if unit["block"] in ("llm_headline", "llm_capped"):
            key = tuple((k, str(v)) for k, v in sorted(unit.items()) if k not in ("arm", "audit", "intervention", "block", "sanitize_mode"))
            groups.setdefault(key, []).append(unit)
    saved = 0
    judge_arms = set(core.get(s, "llm.judge_arms"))
    for members in groups.values():
        first = members[0]
        round0 = first["conversations"] * first["agents"]
        saved += (len(members) - 1) * round0
        saved += max(0, sum(1 for m in members if m["arm"] in judge_arms) - 1) * round0
    return saved


def _usd(calls: dict, s: Mapping) -> float:
    prices = core.get(s, "estimates.usd_per_million_tokens")
    try:
        backbones = list(core.get(s, "llm.backbones"))
    except core.Unset:
        backbones = [m for m in prices if m != core.get(s, "llm.judge.model")]
    agent_in = sum(prices[b][0] for b in backbones) / len(backbones)
    agent_out = sum(prices[b][1] for b in backbones) / len(backbones)
    judge_in = prices[core.get(s, "llm.judge.model")][0]
    agent_calls = calls["agent"] + calls["adversary"] + calls["rewrite"] + calls["ma_judge"]
    return (agent_calls * (core.get(s, "estimates.input_tokens_per_agent_call") * agent_in
                           + core.get(s, "estimates.output_tokens_per_agent_call") * agent_out)
            + calls["judge"] * core.get(s, "estimates.input_tokens_per_judge_call") * judge_in) / 1e6


def _adaptive_line(units, s: Mapping, wiring) -> str:
    """Episodes of the best-response search (simulated before the block itself), or why they cannot be counted."""
    try:
        return f"adaptive search: {wiring.adaptive().search_episodes(units, s):,} episodes"
    except core.Unset as e:
        return f"adaptive search: not sized, set {e.args[0]}"
    except Exception as e:      # feature not built, or no wiring in a bare dry run
        return f"adaptive search: not sized ({e})"


def seconds_per_episode(s: Mapping, arm: str, kind: str, n: int) -> float | None:
    """Measured CPU seconds of one mock episode (estimates.mock_seconds_per_episode_by_arm); an arm without a value takes
    the slowest arm of the same kind and N; None when nothing is measured for N."""
    table = core.get(s, "estimates.mock_seconds_per_episode_by_arm").get(str(n))
    if not table:
        return None
    if f"{arm}|{kind}" in table:
        return table[f"{arm}|{kind}"]
    same = [v for k, v in table.items() if k.endswith(f"|{kind}")]
    return max(same) if same else max(table.values())


def dry_run(s: Mapping, names: List[str], wiring=None) -> None:
    rate = core.get(s, "estimates.calls_per_second")
    workers = core.get(s, "compute.workers")
    print(f"profile: {s['profile']}   compute.workers {workers}\n")
    print(f"{'block':<26}{'units':>10}{'conv/episodes':>15}{'LLM calls':>13}{'USD est':>10}{'cpu h':>9}{'wall h':>9}")
    all_llm: List[dict] = []
    totals = dict.fromkeys(CALL_KINDS, 0)
    for name in names:
        block = BLOCKS[name]
        try:
            units = block.units(s, False)
        except core.Unset as missing:
            print(f"{name:<26}{'scope not set: ' + missing.args[0]:>57}")
            continue
        if block.platform == "llm":
            calls = dict.fromkeys(CALL_KINDS, 0)
            for u in units:
                for k, v in llm_calls(u, s).items():
                    calls[k] += v
            all_llm += units
            for k in totals:
                totals[k] += calls[k]
            n = sum(calls.values())
            print(f"{name:<26}{len(units):>10,}{sum(u['conversations'] for u in units):>15,}{n:>13,}"
                  f"{_usd(calls, s):>10,.0f}{'-':>9}{n / rate / 3600:>9.1f}")
        else:
            groups = units.cost_groups()
            count = sum(groups.values())
            io = core.get(s, "estimates.mock_checkpoint_seconds_per_unit")
            costs = {key: seconds_per_episode(s, key[0], key[1], key[2]) for key in groups}
            known = all(v is not None for v in costs.values())
            cpu = sum(c * (costs[key] + io) for key, c in groups.items()) / 3600 if known else None
            print(f"{name:<26}{count:>10,}{count:>15,}{'0':>13}{'0':>10}"
                  f"{f'{cpu:.2f}' if known else 'n/a':>9}{f'{cpu / workers:.2f}' if known else 'n/a':>9}")
            if block.adaptive:
                print(f"{'':<26}{_adaptive_line(units, s, wiring)}")
    total = sum(totals.values())
    if total:
        saved = cache_savings(all_llm, s)
        print(f"\nLLM calls: {total:,} (agents {totals['agent']:,}, judge {totals['judge']:,} upper bound, "
              f"adversary {totals['adversary']:,}, sanitize rewrites {totals['rewrite']:,}, "
              f"MA judge {totals['ma_judge']:,}); about {total / rate / 3600:.1f} h at {rate} calls/s")
        print(f"with the request cache (round 0 shared across arms): about {total - saved:,} calls, "
              f"{(total - saved) / rate / 3600:.1f} h")
