#!/usr/bin/env python3
"""Small Adaption integration pilot for AntGalasso/swarmlab main.

Default mode is plan: no credentials, API requests, or credit use.
ready: one READY request plus an exact cache check.
pilot: READY, then 6 exploratory episodes on tune seed 9001.
fake: the same integration path, with a local deterministic fake transport.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SOURCE_COMMIT = "a94ebe2fbabb6094254d2b3b34306ae000d580b1"
RATES = {  # Adaption credits per million tokens; verified 2026-11-06.
    "gemma-4-31b-it": (0.09, 0.34),
    "deepseek-v4-flash": (0.04444, 0.13332),
    "qwen3.8-27b": (0.039, 1.25),
}
ARMS = ["no_defense", "static_zta", "bazt"]
MAX_CALLS = 865                 # one READY + 6 episodes x 12 agents x 12 steps
MAX_INPUT_BYTES = 8192
MAX_OUTPUT_TOKENS = 512


class PilotError(RuntimeError):
    pass


class Meter:
    def __init__(self, model, limit, fake=False):
        self.model, self.limit, self.fake = model, limit, fake
        self.calls = 0
        self.accounted_credits = 0.0
        self.usage_credits = 0.0
        self.events = []

    def reserve(self, messages, max_tokens):
        size = sum(len(m["content"].encode("utf-8")) for m in messages)
        if size > MAX_INPUT_BYTES:
            raise PilotError("Prompt exceeds the pilot's 8192-byte limit; review the scope before enlarging it.")
        if self.calls >= MAX_CALLS:
            raise PilotError("Pilot request limit reached.")
        pin, pout = RATES[self.model]
        # Conservative planning proxy, NOT a provider-side billing guarantee.
        reserve = ((size + 256) * pin + max_tokens * pout) / 1_000_000
        if self.accounted_credits + reserve > self.limit:
            raise PilotError("Client-side credit allowance reached; no further request was sent.")
        self.calls += 1
        self.accounted_credits += reserve
        return reserve

    def finish(self, reserve, pt, ct):
        pin, pout = RATES[self.model]
        actual = (pt * pin + ct * pout) / 1_000_000
        self.accounted_credits += actual - reserve
        self.usage_credits += actual
        self.events.append({"call": self.calls, "input_tokens": pt, "output_tokens": ct,
                            "credits_at_documented_rates": actual})
        if self.accounted_credits > self.limit:
            raise PilotError("Reported usage exceeded the planning reserve; stopping before another request.")

    def summary(self):
        return {"network_requests": 0 if self.fake else self.calls,
                "fake_requests": self.calls if self.fake else 0,
                "input_tokens": sum(x["input_tokens"] for x in self.events),
                "output_tokens": sum(x["output_tokens"] for x in self.events),
                "credits_from_reported_usage": 0.0 if self.fake else self.usage_credits,
                "allowance_credits": self.limit,
                "reserve_accounting": self.accounted_credits,
                "note": "Client allowance uses a conservative token proxy and published rates; verify actual billing in Adaption."}


def source_state(repo):
    hashes = {}
    for name in ("swarmlab.py", "llm_client.py", "bazt/runner.py", "bazt/config.py", "bazt/arms.py"):
        path = repo / name
        if not path.is_file():
            raise PilotError("Missing repository file: " + name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    manifest = repo / "UPSTREAM_SOURCE.json"
    if manifest.is_file():
        expected = json.loads(manifest.read_text(encoding="utf-8"))["verified_runtime_source_sha256"]
        if hashes != expected:
            raise PilotError("Runtime source differs from the verified private snapshot; validate changes and update UPSTREAM_SOURCE.json before running.")
    elif head is not None:
        descendant = subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
                                    cwd=repo, capture_output=True, check=False).returncode == 0
        if not descendant:
            raise PilotError("This pilot requires main " + SOURCE_COMMIT + " or a descendant checkout with full history.")
    return {"upstream_base_commit": SOURCE_COMMIT, "checkout_head": head, "source_sha256": hashes}


def run(args):
    repo = Path(args.repo).expanduser().resolve()
    state = source_state(repo)
    pin, pout = RATES[args.model]
    plan = {"mode": args.mode, "provider": "adaption", "model": args.model,
            "episodes": 6, "topologies": ["small_world"], "regimes": ["clean", "decoy"],
            "arms": ARMS, "tune_seed": 9001, "agents": 12, "steps": 12,
            "max_requests_including_READY": MAX_CALLS, "concurrency": 2, "retries": 0,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "max_prompt_bytes": MAX_INPUT_BYTES,
            "published_credits_per_million_tokens": {"input": pin, "output": pout},
            "conservative_planning_credits": MAX_CALLS * ((MAX_INPUT_BYTES + 256) * pin + MAX_OUTPUT_TOKENS * pout) / 1_000_000,
            "allowance_credits": args.max_credits,
            "scientific_status": "Exploratory integration pilot with untuned defaults; not a confirmatory result.",
            **state}
    if args.mode == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    try:
        from dotenv import load_dotenv
        load_dotenv(repo / ".env", override=False)
    except ImportError:
        pass
    live = args.mode in ("ready", "pilot")
    if live and not os.getenv("ADAPTION_API_KEY"):
        if not args.prompt_key:
            raise PilotError("ADAPTION_API_KEY is not set. Put it in the local .env, or use --prompt-key for hidden input. Never paste it into chat.")
        key = getpass.getpass("Adaption API key (hidden, not saved): ").strip()
        if not key:
            raise PilotError("No key supplied; no request sent.")
        os.environ["ADAPTION_API_KEY"] = key
    os.environ["SWARMLAB_PROVIDER"] = "adaption"
    import bazt.runner as br
    import swarmlab as sl
    from bazt.arms import RunContext
    from bazt.config import BaztConfig
    from llm_client import LLMBackend, SQLiteLLMCache
    if args.mode == "pilot":
        sl._kickoff_guard(SimpleNamespace(mode="llm", allow_live_llm=args.allow_live_llm))
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(args.out).expanduser().resolve() if args.out else repo / "results" / ("ricardo_adaption_" + stamp)
    if out.exists():
        raise PilotError("Output directory already exists; choose a new one to preserve the previous run.")
    out.mkdir(parents=True)
    meter = Meter(args.model, args.max_credits, fake=args.mode == "fake")
    api_client = None
    if live:
        from adaption import AsyncAdaption
        api_client = AsyncAdaption(api_key=os.environ["ADAPTION_API_KEY"], timeout=60.0, max_retries=0)

    async def transport(model, messages, temperature, max_tokens):
        reserve = meter.reserve(messages, max_tokens)
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if args.mode == "fake":
            if system == "Reply with the single word READY.":
                text = "READY"
            else:
                text = "est=" + (re.search(r"current estimate: (-?[\d.]+)", rest[0]["content"]).group(1))
                if "[REF-7]" in system + rest[0]["content"]:
                    text += " [REF-7]"
                if "[DOC-2]" in system:
                    text = "[DOC-2] " + text
            pt = (sum(len(m["content"].encode()) for m in messages) + 3) // 4
            ct = max(1, (len(text) + 3) // 4)
        else:
            try:
                response = await api_client.responses.create(
                    model=model, instructions=system, input=rest, temperature=temperature,
                    max_output_tokens=max_tokens, store=False)
            except Exception as exc:
                raise PilotError("Adaption request failed: " + type(exc).__name__ +
                                 "; HTTP " + str(getattr(exc, "status_code", "unknown")) +
                                 ". Check account access and hosted-model availability.") from None
            usage = getattr(response, "usage", None)
            if usage is None or getattr(usage, "input_tokens", None) is None or getattr(usage, "output_tokens", None) is None:
                raise PilotError("Response has no complete token usage; stopping without further calls.")
            pt, ct = int(usage.input_tokens), int(usage.output_tokens)
            meter.finish(reserve, pt, ct)
            text = (getattr(response, "output_text", "") or "").strip()
            if getattr(response, "error", None) or getattr(response, "status", "completed") != "completed":
                raise PilotError("Adaption response did not complete; no experimental row accepted.")
            if not text:
                raise PilotError("Adaption returned empty output; no experimental row accepted. Review the output-token limit/model.")
            return text, pt, ct
        meter.finish(reserve, pt, ct)
        return text, pt, ct

    cache = SQLiteLLMCache(str(out / "request_cache.sqlite"))
    backend = LLMBackend(args.model, cache, transport=transport, max_concurrency=2,
                         temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS, retries=0)
    receipt = {"plan": plan, "started_utc": dt.datetime.now(dt.UTC).isoformat(),
               "status": "started", "output_directory": str(out)}
    try:
        req = [("Reply with the single word READY.", "Ping.", "ricardo-ready-20261106")]
        first = backend.generate(req)[0]
        calls_after_first = meter.calls
        second = backend.generate(req)[0]
        if first.strip() != "READY" or second != first or meter.calls != calls_after_first:
            raise PilotError("READY or exact cache check failed; pilot not started.")
        receipt["ready_check"] = "passed"
        print("READY and exact cache check passed.")
        if args.mode in ("fake", "pilot"):
            cfg = sl.Config()
            context = RunContext(bcfg=BaztConfig(),
                                 agent_factory=lambda c, w: sl.LLMAgentPolicy(backend), equalize=True)
            rows, audit = br.run_bazt_grid(["small_world"], ["clean", "decoy"], ARMS, [9001],
                                         cfg, context, collect_audit=True)
            br.finalize(rows)
            br.write_rows(rows, str(out / "pilot.csv"))
            (out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
            if len(rows) != 6:
                raise PilotError("Pilot produced an unexpected number of rows.")
            receipt["episodes_completed"] = len(rows)
            print("6 exploratory episodes completed; mock/fake output is never a scientific finding." if args.mode == "fake"
                  else "6 live exploratory episodes completed; more seeds and frozen calibration are needed for scientific inference.")
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error_type"] = type(exc).__name__
        raise
    finally:
        receipt["meter"] = meter.summary()
        receipt["cache"] = cache.stats()
        receipt["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        (out / "token_usage.json").write_text(json.dumps(meter.events, indent=2), encoding="utf-8")
        if api_client is not None:
            backend._loop.run_until_complete(api_client.close())
        backend.close()
        cache.close()
        print(json.dumps(receipt["meter"], indent=2))
        print("Results:", out)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Path to the isolated swarmlab main checkout")
    parser.add_argument("--mode", choices=["plan", "ready", "fake", "pilot"], default="plan")
    parser.add_argument("--model", choices=list(RATES), default="gemma-4-31b-it")
    parser.add_argument("--max-credits", type=float, default=1.0)
    parser.add_argument("--prompt-key", action="store_true", help="Read the API key locally without echoing/saving it")
    parser.add_argument("--allow-live-llm", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    if not math.isfinite(args.max_credits) or args.max_credits <= 0:
        parser.error("--max-credits must be finite and positive")
    try:
        return run(args)
    except (PilotError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
