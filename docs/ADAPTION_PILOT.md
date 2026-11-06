# Small Adaption integration pilot

The `main` transport already supports Adaption. The READY check now selects Adaption model IDs when
`SWARMLAB_PROVIDER=adaption`; the OpenRouter defaults remain unchanged. A cached empty or incorrect reply fails
the check. Cache verification uses hit/live-call counters rather than a scheduling-sensitive latency threshold.

```bash
SWARMLAB_PROVIDER=adaption python llm_client.py
python scripts/run_adaption_pilot.py --mode plan
python scripts/run_adaption_pilot.py --mode fake
python scripts/run_adaption_pilot.py --mode pilot --allow-live-llm
```

Set `ADAPTION_API_KEY` in the environment or local `.env`. `--prompt-key` is available for hidden terminal input.
Never add credentials to a commit. Account-specific smoke IDs can be supplied through
`ADAPTION_SMOKE_PRIMARY_MODEL`, `ADAPTION_SMOKE_FALLBACK_MODEL`, and `ADAPTION_SMOKE_H4_MODEL`.

The pilot first requires literal READY and an exact cache hit, then runs `no_defense`, `static_zta`, and `bazt`
on `small_world`, clean/decoy regimes, tune seed 9001, with 12 agents and 12 steps. It uses a fresh output directory,
concurrency 2, no retries, at most 865 requests including READY, and at most 512 output tokens per request. It
records token usage and stops on empty/incomplete output or missing usage. Its client allowance defaults to one
Adaption credit using published rates and conservative reservations; this is not a provider-enforced billing cap.
The documented kickoff guard is preserved. Existing evaluation outputs and frozen settings are not changed.

This is an **integration check with untuned defaults**, not a confirmatory experiment. The LLM writes status
messages; numerical consensus and the BAZT auditor remain simulator-side. Static ZTA's always-on schema policy is
an architectural reference, not an adaptively budgeted arm. Do not interpret zero mock violations as evidence of
real-model safety. Tune/calibrate and freeze on separate seeds before evaluation.

## CI execution

The `Adaption integration pilot` workflow runs offline tests, a fake pilot, and the requested live pilot on pushes
to `ricardo/adaption-pilot-20261106`. For live execution it reads the existing repository secret
`ADAPTION_API_KEY`. It never prints the secret or saves it in artifacts. If the secret is absent, the live step
fails with `blocked_missing_api_key` and sends zero requests; offline evidence is still uploaded. Manual dispatch
defaults to offline-only unless `live_pilot` is selected. Receipts, CSVs, and tests are uploaded as an Actions artifact.

The new `bazt_lab` platform on branch `bazt` has separate OpenRouter agent/judge adapters and is outside this
transport change. Its log-probability judge needs separate validation before it can use an Adaption backend.
