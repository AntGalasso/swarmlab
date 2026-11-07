# Ricardo execution status — verified 7 November 2026

## Integration and protocol

PRs #5, #6 and #7 are merged into their target branches. PR #8 consolidates
the team code into `main` at commit
`2b578d803659f90e0d3f58d275e6704b313ea318`. The integrated Python 3.11 CPU
suite completes all 3,740 collected tests: 3,705 pass, 35 skip and none fail.
Ruff, byte-compilation, the architecture boundary, mock integration and fake
study also pass. GitHub Actions run:
https://github.com/AntGalasso/swarmlab/actions/runs/42305786319.

The Gemma-4-26B-A4B-IT study was frozen before evaluation. The source,
controller settings, cache lineage, analysis implementation, hypotheses,
comparison family and MAE non-inferiority margin were not changed after
outcomes were observed. Gemma-31B attempts failed operationally and are retained
as separate receipts; their outputs are not pooled with Gemma-26B.

## Completed live study

The relay calibration completed 20 episodes. All ten attack-free controls had
zero tracked markers. Mean untreated propagation AUC was 0.4444 on small-world
graphs and 0.9236 on star-hub graphs, so the pre-specified challenge gate passed.

The frozen evaluation then completed the full grid:

- main: 240 unique episodes, seeds 1--12;
- held-out: 120 unique episodes, seeds 31--36;
- two topologies, two regimes and five policies;
- zero duplicate grid cells and zero enforcement-cap violations.

The full private reproduction archive is
`results/ricardo/integrated_study_v6/completed_reproduction.zip`, SHA-256
`a98619fbec952afff474fbd1636e9ac1ffa8a07c221d5a6965b6ee19205fe580`.
The frozen analysis SHA-256 begins `938eef64`. The successful evaluation and
finalization workflows are:

- https://github.com/ricardofitas/swarm-adaption-hackathon/actions/runs/42316134013
- https://github.com/ricardofitas/swarm-adaption-hackathon/actions/runs/42325018863

## Scientific result

H2 is unsupported. In the main analysis, the score-only decoy-minus-clean
propagation contrast is 0.0168 (95% CI [-0.0014, 0.0370], one-sided p=0.0938).
The held-out estimate reverses sign: -0.0122 (95% CI [-0.0295, 0.0023],
p=0.875).

H3 receives joint Holm-adjusted safety--utility support only in the main clean
regime:

- versus score-only: delta AUC -0.5506 and delta MAE -1.1009;
- versus centrality-only: delta AUC -0.0773 and delta MAE -4.7515.

Under decoys, BAZT is safer than score-only but misses the 0.5-unit utility
margin by a wide amount (delta MAE +3.1977). It is worse than centrality-only
and the no-pacing ablation on both observed endpoints. The six-seed held-out
design cannot achieve Holm-adjusted p<0.05 within the frozen 12-test family;
its estimates are therefore reported as an independent descriptive consistency
check, not pooled with main and not treated as proof of absence.

Action traces support the proposed budget-timing mechanism but not an endpoint
advantage. In main decoy episodes, paced BAZT exhausts the cap in 5/24 episodes
at mean step 8.4, whereas no-pacing exhausts it in 23/24 at step 1. The same
pattern appears held out (4/12 versus 9/12). Pacing preserves later capacity,
but no-pacing has better final decoy AUC and MAE in this campaign.

## Interpretation and deliverables

The result is mixed and narrower than the original hypothesis. BAZT is useful
in the clean simulator condition against two comparators, but the decoy regime
exposes a safety--utility failure. The evidence concerns simulator-supplied
numerical consensus, a marker-based parametric monitor and a simulator-side
inspection oracle. It does not establish general LLM safety, a learned auditor,
or deployment-level robustness. No no-topology/no-provenance ablation or H4
cross-model fidelity experiment is reported as completed.

A separate prospective Early/K=3 and accessible-model H4 extension is now
running on new seeds. Its source, analysis and configuration were frozen
before inference; old v6 outcomes are not pooled with it. The primary model
is Gemma-4-26B-A4B-IT and the matched second model is Qwen3.8-27B. This is
not the original Qwen-2.5-7B/GPT-4o-mini contrast, which remains untested with
the configured credentials. The no-pacing extension holds theta0 and kappa
identical to full BAZT. See the dedicated execution and protocol:

- https://github.com/ricardofitas/swarm-adaption-hackathon/actions/runs/42351694248
- https://github.com/ricardofitas/swarm-adaption-hackathon/blob/ricardo/h2h4-20261107/docs/H2H4_EXTENSION_20261107.md

The v6 "no-pacing" comparison above used separately tuned thresholds and is
therefore a comparison of tuned configurations; it does not by itself isolate
the causal effect of pacing.

The private Overleaf manuscript was updated, compiled and visually checked.
Its body is eight pages, followed by one reference page and a two-page appendix.
Team authorship and affiliations remain pending. The project has not been
submitted and no third-party message has been sent.
