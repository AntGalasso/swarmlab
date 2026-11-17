# Review follow-up: BAZT paper v2

This note turns the sprint feedback into a paper-ready summary, a consolidated parameter table, and a rerun plan. Review reruns and a two-seed live pilot are complete. The mock outputs are pipeline/exploratory checks, not confirmatory findings; the live pilot returned NO-GO. A paste-ready manuscript revision packet is saved separately in the outputs folder.

## Plain-language summary

BAZT is a gateway that checks messages before they pass between AI agents. It combines a risk score with the message's path through the network and an estimate of how far it could spread. With separate budgets for checks and interventions, it can choose a graduated response: remove free text, restrict an agent, block one connection, or isolate a sender. The Stop rule skips a paid check when the remaining budget cannot fund an intervention. It is an opt-in setting in the active optimizer; the standard `bazt` arm enables it and `bazt_no_stop` disables it for comparison. Six focused regression tests pass. Exploratory mock sweeps show reduced audit spending under tight intervention caps, with identical measured outcomes in tested cells; Stop does not activate at the 10%/10% cap. The two-seed live pilot returned NO-GO against contextual Zero Trust, so no superiority claim is supported.

## Parameters reported in paper v2

These are the values stated in the submitted paper, not a claim that each value matches current repository defaults. Reconcile this table with the corrected frozen configuration before reusing it in a revised paper.

| Parameter | Paper-v2 value | Meaning |
|---|---:|---|
| Agents, horizon, task | `N = 12`, `T = 12`; readings `N(50, 10²)`; accuracy tolerance `3.0` | Number of agents, communication rounds, and numerical consensus task settings reported in the paper. |
| Judge-score window | `m = 2` | Rolling window used to smooth judge scores. |
| Inspection | Cost `1` audit unit; verdict lifetime `2` steps; `q = 0.8` | Paid inspection and its reported accuracy and duration. |
| Provenance taint | ingress `1.0`; per-hop `0.7`; per-step decay `0.8` | Structural signal propagated along delivered messages. |
| Evidence fusion | `a = 2.5`, `b = 2.0`, `w = 1.0` | Logistic fusion of the rolling score and provenance taint. |
| Reach estimate | `γ = 0.7`; `d_max = 3` | Path decay and maximum look-ahead distance. |
| Sanitize | Cost `1`; `ρ_san = 0.1` in paper v2 | Removes free text while retaining the structured numeric value. Active `BaztConfig` currently says `ρ_san = 0.15`. |
| Restrict | Cost `2`; duration `3` steps; `ρ_res = 0.3` in paper v2 | Enforces schema-only output and denies tools for the reported duration. Active `BaztConfig` currently says `ρ_res = 0.35`. |
| Block edge / isolate | Costs `2` / `3` | Drops one directed delivery / severs all outgoing edges from a sender. |
| Pacing threshold | `θ₀ = 0.150`; `κ = 1.5` | Exponential rise in the minimum action-value density as the intervention budget is spent. |
| Utility weights | `λ = 1.0` for the gateway; `λ = 0.25` for the containment-suite policy | Keep these contexts separate; they are not one shared setting. |
| Graduated containment | `τ_san = 0.35`; `τ_quar = 2.0`; `K = 3`; `B_san = 6` | Reported sanitization and quarantine settings for the two-tier policy. |
| Budget settings | Gateway pilot: audit `10%`, intervention `10%`; Stop simulator: `10%` exposure cap; Stop live study: audit `20%`, intervention `1%` / `5%` | Keep the simulator exposure cap distinct from the live audit and intervention caps. |
| Graph | BA `m = 2`; Watts–Strogatz `k = 4`, `p = 0.25` | Main graph-generation settings reported in the paper. |
| Attack and decoys | `β = 0.35`; persistence `0.80`; numeric bias `8.0`; 3 decoy agents with burst probability `0.90` | Simulator infection and benign false-alarm settings reported in the paper. |
| Stop price floor | smallest eligible positive action cost `c⁺_min` | In the active optimizer, the check is recalculated before each proposed inspection. Pooled and tiered modes reserve the inspection's cost in its pool; `dual_pct` uses the separate intervention pool. Current `BaztConfig` requires all action costs to be positive, so a free-action exception is not currently representable. The reported `stop_candidates_suppressed` counter is a count of proposals clearing the current threshold when Stop activates, not a no-Stop counterfactual purchase count. |

## Evidence corrections and limits

- The merged [PR #7](https://github.com/AntGalasso/swarmlab/pull/7) reports that the shipped `arm_C` calibration used `rl_offset = -0.6`, leaving it too permissive and inactive in the comparison; it also corrected cross-stage statistical pairing. Audit all pre-PR quantitative claims, not just Table 1. The abstract, Table 1, conclusion, and two-episode DeepSeek pilot repeat BAZT-versus-`arm_C` comparisons. Recompute affected estimates and intervals from corrected results; do not merely change the displayed interval.
- The repository labels [`results/experiments/paper_pack/STATUS.md`](../results/experiments/paper_pack/STATUS.md) as superseded mock verification output. Its old numerical gains and inferential labels must not be used as findings; a full corrected semantic-task evaluation remains outstanding.
- The paper's mechanism ablation has two seeds per cell. Describe its differences as preliminary; do not claim causal attribution from those cells.
- The qwen live sweep's result depends on one outbreak seed. The DeepSeek pilot has two episodes. The live Stop study used clean worlds without an attacker. Keep all three limitations explicit unless new evaluations address them.

## Recommended evaluation sequence

1. **Inventory and freeze.** Build a claim-by-claim inventory for H1–H3, gateway comparisons, ablations, and every live result. Mark each as unaffected, needs reanalysis, or needs a new run. Reproduce corrected `arm_C` calibration, record configuration hashes, verify the baseline intervenes, and freeze the analysis plan before rerunning.
2. **Rerun affected gateway comparisons.** Use corrected configurations and paired common-random-number cells. Report the resampling unit and bootstrap method, number of independent seeds and cells, effect estimates, intervals, and non-inferiority result. Replace every dependent number in the paper, README, figures, and abstract together.
3. **Strengthen the graduated-action ablation.** Expand from two to at least 30 fresh paired simulator seeds per topology, keeping all other settings fixed. Report per-topology paired estimates and intervals; retain the current two-seed values only as pilot results.
4. **Validate Stop, then strengthen the budget-reactive decoy evaluation.** Completed on the review branch: the Stop/no-Stop arm, public-state budget telemetry, regression coverage for pooled/tiered/dual-pool funding boundaries, matched simulations, and separate persistent, pressure, ledger-unfundable, and fixed-round decoy schedules. The ledger-unfundable schedule uses a coarse public-ledger feasibility proxy, not the optimizer's exact per-message Stop condition. Five exploratory 30-seed sweeps (1%, 5%, 10%, and pressure thresholds 0.25/0.75 at 1%) produced 1,500 rows; see `outputs/STOP_TRIGGER_SWEEP_SUMMARY_2026-11-17.md`. The sweeps found no measured outcome difference between Stop and no-Stop, while audit savings were concentrated at tight caps. These public-budget attacker checks are outside the core threat model, which excludes attackers who observe budget state. The separate best-response search remains distinct and has not been validated as reacting to public gateway state. A stronger adaptive attacker evaluation and live attacked worlds remain future work.
5. **Run attacked live Stop worlds last.** After regression coverage and simulator evaluation, pair Stop and no-Stop on the same prompts, model settings, and independent worlds, with an attacker scheduling decoys around the visible Stop point. Set sample size and cost after pre-registration. Report uncertainty over worlds, final-answer disagreement, infection, audit/token cost, and utility. Do not describe the existing clean-world test as adversarial validation.
6. **Manuscript length and claims.** The attached paper already has an eight-page main text, so do not shorten it further. Replace unsupported claims in place, reconcile the main-text parameter table against frozen run configs, preserve a concise limitations paragraph separating mock from live evidence, and repair the stale appendix row and missing figure placeholders. A paste-ready revision packet is available as `outputs/BAZT_paper_v2_revision_packet_2026-11-17.md`.

## Paper-ready parameter-table note

Before publication, add a source column or configuration identifier to the table above and verify every value against the exact frozen run configuration. The paper values for `ρ_san` and `ρ_res` differ from the active config defaults. If a value differs between paper v2 and the corrected implementation, report the value actually used in each result rather than silently harmonizing them. The table is a working inventory, not publication-ready until that reconciliation is complete.
