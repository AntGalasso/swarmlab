# analysis: contract

The statistics of `EXPERIMENT_PLAN.md` section 7 over the rows the runner writes to
`results/experiments/<profile>/<block>/units/` and its worker shards `units/w<k>/` (worker k). It reads rows and
writes one JSON report. It runs no episode, draws no experiment random number except the seeded bootstrap, and reads no
external service.

## public.py

| name | what |
|---|---|
| `analyze(results_dir, settings, out=None) -> dict` | runs every analysis on both platforms, writes the report to `out` (default `<results_dir>/analysis_report.json`, replaced atomically) and returns it as written. `results_dir` is the folder that holds the block folders (`results/experiments/<profile>`) |
| `main(argv)` | the command line, `python -m bazt_lab.features.analysis` (below); returns the exit code, 0 or 2 |
| `missing_keys(settings) -> list[str]` | every setting the analysis needs that is not set |
| `REPORT_NAME` | `"analysis_report.json"` |

`analyze` raises `core.Unset` naming the first unset setting before it reads anything, `FileNotFoundError` for a
`results_dir` that is not a directory, `ValueError` for an invalid setting, a malformed row, a row whose `attack_free`
contradicts its `attack`, or a repeated episode inside one group.

Command line: `python -m bazt_lab.features.analysis [--profile P] [--set key=value ...] [--settings FILE]
[--results-root DIR] [--out FILE]`. It reads `<results-root>/<profile>`, writes the report, prints the verdicts and
exits 0. A refusal (every unset setting listed, a missing directory, malformed rows) prints `Refusing analysis: ...` to
stderr, writes nothing and exits 2.

## Rows read

Files: the committed rows files of `<block>/units/` and of every worker shard `<block>/units/w<k>/`
(`core.CheckpointStore(shard=...)`, one shard per worker process when `compute.workers > 1`), whatever worker count each
launch of the block used, as `core.committed_row_files` lists them. A rows file `<stem>.jsonl` is committed when
`<stem>.done` is in the same folder, the store's own done rule: the done list of a batch (mock) or the done marker of a
unit (LLM). A rows file without it is not read
and is counted in `meta.platforms.<p>.uncommitted_files`: a batch a crash cut before its done list (an orphan; the
store clears orphans only in its own write folder, so one in a shard no later launch writes to stays on disk) or a unit
still running, or abandoned in one shard and run again in another. A listed block with no committed row is in
`meta.platforms.<p>.blocks_missing`, like a block that is not on disk.

One JSON object per line, as `metrics.rows_from_trace` writes them: the unit fields flat in the row, or nested in a
`"unit"` object (the row wins a clash). A torn line is skipped and counted.

| fields | use |
|---|---|
| `arm`, `audit`, `intervention` (fraction of deliveries or `"uncapped"`), `topology`, `agents`, `condition`, `preset`, `cost_condition`, `sanitize_mode`, `ma_check` (D101, D103: a research-profile episode is repeated once per value, so each value is its own group and its own comparison. An attack-free run carries the same `ma_check` as its attacked rows (rule with rule, llm_judge with llm_judge; None on both sides for non-MA tasks), and the clean twin matches `sanitize_mode` and `ma_check`. An arm that never sanitizes runs once, in the first mode, and serves every `sanitize_mode` of the other arm: comparisons, containment, efficiency and the adaptive reference resolve a missing group to the single group that differs only in `sanitize_mode`) | the cell. Two arms are compared inside the same cell of the same block |
| `attack` (`"none"` for attack-free runs), `seed`, `sink`, `conversation`, `task`, `backbone` | the episode, the pairing key between two arms. An LLM row needs `conversation` |
| `accuracy_by_round`, `accuracy_clean` (LLM) | the clean pairing and the recovery time (below) |
| `asr_final` | safety endpoint (spec section 7), attacked rows |
| `accuracy_clean` or `accuracy_attack` (`analysis.utility_endpoint`) | utility endpoint |
| `spend_audit_total`, `spend_intervention_total` | spend, containment per cost, frontier budgets |
| every column named in `analysis.extra_metrics` and `analysis.containment_metrics` | extra endpoints, gain per cost |

A missing metric column reads as NaN and is listed in `meta.platforms.<p>.empty_metrics` when no row has it.

A group is the set of rows of one block, arm, budget cell (audit, intervention), topology, agent count, condition,
preset, cost condition, sanitize mode, MA check and attack-free flag. Within a group an episode is (attack, seed, sink,
conversation, task, backbone), and two groups are paired on the episodes they share. An episode repeated inside one
group raises `DuplicateRows`, a `ValueError` that names the first repeat.

## Statistics

- Pairing. Mock: the pair is the seed, so a continuous metric is averaged over the sinks and families of a seed first
  (`analysis.pair_unit.mock = "seed"`). LLM: the pair is the conversation (`"episode"`). A binary metric always pairs
  episodes. A pair with a NaN side is dropped and counted.
- Continuous metrics: two-sided paired Wilcoxon signed-rank (`stats_check.wilcoxon_rrb`) and a paired bootstrap CI of
  the mean difference, `analysis.bootstrap_resamples` resamples (1,000 in the spec). The bootstrap resamples whole seeds (clusters): a resample draws
  seeds with replacement and takes every pair of a drawn seed. Draws come from `core.stream("analysis", "bootstrap",
  <comparison key>)`, one `[resamples, clusters]` block per comparison, so a report depends on its rows and settings only,
  never on row order, file split or call order.
- Binary metrics: McNemar's exact test (exact two-sided binomial on the discordant pairs), same cluster bootstrap CI.
- Every paired test reports `se_ratio_cluster_vs_naive`, the cluster-bootstrap standard error over the naive one.
  Above 1 the pairs are positively correlated inside a seed and the Wilcoxon p-value is anti-conservative; the interval
  is not.
- Holm (`stats_check.holm`): one family per hypothesis, platform, challenger and endpoint, over every comparison the
  hypothesis makes there (budget cells, topologies, agent counts, and attack families, tasks or backbones when
  `analysis.split_by` names them). Safety, utility and each extra metric are adjusted separately. The ladder and the
  ablations are one family each per platform, over all their steps or all their arms.
- Labels (`stats_check.pareto_label`, through `adapters/stats_check_adapter.py`, the labels of D43): safety endpoint
  `asr_final` at the final round (`delta < 0` is the challenger safer); utility endpoint accuracy expressed as a loss
  (baseline minus challenger), so `STRICT_PARETO_WIN`, `SAFETY_WIN_UTILITY_NONINFERIOR` (upper end of the
  `analysis.ci_level` interval of the loss <= `runtime.ni_margin_accuracy`), `UTILITY_LOSS`,
  `CONTAINMENT_GAIN_UTILITY_UNBOUNDED`, `NO_SIGNIFICANT_DIFFERENCE`. The significance level is `stats_check.ALPHA`;
  `analysis.alpha` is refused when it differs.
- A hypothesis is supported when at least half its comparisons are `STRICT_PARETO_WIN` or
  `SAFETY_WIN_UTILITY_NONINFERIOR` and none is `UTILITY_LOSS` (D43). `summary.complete` is false when a cell could not
  be compared; the cell and the reason are in `skipped`.

## Clean pairing (LLM) and insider sinks

The LLM platform cannot pair an attack row with its attack-free run (a different unit), so its rows carry NaN
`accuracy_clean_reference`, `nrp` and `recovery_time_accuracy`. Analysis forms them on the LLM table (D88, D89): each
attack row is paired with the attack-free row of the same block, arm, budget cell, topology, agents, condition, preset,
cost condition, sanitize mode, MA check, backbone, task, seed, sink and conversation; `accuracy_clean_reference` = the
twin's `accuracy_clean`; `nrp` = reference x (1 - `asr_final`);
`recovery_time_accuracy` = `metrics.public.recovery_time_accuracy(accuracy_by_round, reference, runtime.recovery_margin)`
(NaN when the row has no `accuracy_by_round`). A row whose reference is already set is left as the platform wrote it. An
attack row without a twin is a `ValueError` naming the unit, never a silent drop. Mock rows are not touched. The clean accuracy of a comparison (utility endpoint, attack-free extra metrics) is taken from the attack-free rows of the tasks and backbones of the comparison's own attacked rows, never from another task.

D90: metrics gives `protected_sink_harm` NaN where the sink is an insider. Every paired test drops NaN pairs (counted in
that test's `n_dropped_nan`), so those episodes never enter an average; `meta.platforms.<p>.sink_attacker_excluded` counts
the attacked rows with `sink_is_attacker` = 1.

H-adaptive details. An adaptive block (`llm_adaptive_check`) may have no attack-free units: its utility endpoint (and
its clean twin for NRP) then comes from the attack-free rows of the `adaptive_reference` block, same arms and episodes.
The clean twin of an adaptive LLM row is the attack-free run of the same unit with no `condition`, in any loaded block.
The runner builds the adaptive units on the seeds, topologies and agent counts of their reference block (`mock_adaptive`
on `mock_frontier`, `llm_adaptive_check` on `llm_headline`), so the episodes pair.
Rows of an adaptive block without a `condition` are listed in `skipped` with the reason and make the hypothesis
`complete: false`.

## Analyses

| analysis | what it computes |
|---|---|
| H-budget | challenger against the baseline (capped C) in every cell where a budget is capped; one comparison per cell |
| H-efficiency (descriptive) | per challenger and site (block, topology, agent count and the other unit fields) and split combination: a cell reaches when its mean ASR minus the uncapped baseline's is at most `analysis.efficiency_tolerance`; the smallest budget (audit + intervention fractions) among reaching cells, its bootstrap interval, `p_reached`, and the Pareto-minimal reaching cells. One seed resample is applied to every cell of the frontier |
| H-topology | the H-budget comparisons grouped into `analysis.topology_classes`; supported when the advantage appears on heterogeneous topologies (class rule), is absent on homogeneous ones (class rule fails) and grows with N (Mann-Whitney U on the advantage at the largest against the smallest agent count, per heterogeneous cell, Holm across cells; growth holds when at least half the cells grow significantly and none shrinks). A part the data cannot speak to is left out and listed in `unevaluated` |
| H-adaptive | per knowledge condition and cell, `d = delta_challenger - factor * delta_baseline` with `delta` = ASR under the adaptive attacker minus ASR of the same episodes in the non-adaptive reference block (`analysis.blocks.adaptive_reference`); `factor` is set by `analysis.adaptive_contrast` |
| ladder | every rung against the one below (`analysis.ladder`), uncapped cells; a cell is ordered when every step is a significant safety gain |
| ablations | `analysis.ablation_reference` (the full arm) is the challenger against each `analysis.ablations` arm (the baseline), one Holm family per platform; a win means the removed component helps |
| rank agreement | the pre-registered arm pairs ranked by mean final-round ASR on each platform (uncapped attacked runs, cells where every ranked arm ran, mean within seed then over cells then over seeds); Spearman or Kendall (`analysis.rank_agreement_statistic`) with a seed-bootstrap interval; supported when the value is at least `runtime.rank_agreement_threshold`; a tie or a missing arm gives `not_evaluable` or an undefined statistic that is not supported |
| containment per cost | for each arm and cell, `sum(reference - arm) / sum(spend)` over the paired episodes for every `analysis.containment_metrics` column, the reference being the uncapped `analysis.containment_reference` run (no budget cell), looked up in the arm's own block and then in every block listed in `analysis.blocks.containment.<platform>` (A runs only in the headline block), matching the episode fields; NaN when nothing was spent |

## Report

```
{"meta": {profile, results_dir, pre_registered {ni_margin_accuracy, rank_agreement_threshold}, settings_sha256,
          analysis_settings, manifests {checked, pre_registered_mismatch [...]},
          platforms {mock|llm: {rows, blocks, blocks_missing, torn_lines, uncommitted_files, empty_metrics,
                                sink_attacker_excluded}}},
 "hypotheses": {"H-budget"|"H-efficiency"|"H-topology"|"H-adaptive": {mock|llm: result}},
 "secondary": {"ladder": {mock|llm: result}, "ablations": {mock|llm: result}, "rank_agreement": result},
 "containment_per_cost": {mock|llm: result},
 "verdicts": {"H-budget"|"H-topology"|"H-adaptive": {mock|llm: {challenger: bool} | status}}}
```

A platform result is `{"status": "not_configured"}` (no block listed in `analysis.blocks`), `{"status": "no_data",
"blocks_missing": [...]}` (no listed block has a committed row on disk) or `{"status": "ok", ...}`. `rank_agreement` is `{"status": "ok", ...}` or `{"status": "not_evaluable", "reason"}` (no rank
block, or no cell where every ranked arm has uncapped attacked runs). Every number is finite or `null`; the file
is strict JSON. `meta.manifests` compares the two pre-registered values with the `pre_registered` entry of every
`manifest-*.json` in the blocks read and lists each difference (and each unreadable manifest); it never changes a value.
`meta.settings_sha256` fingerprints the analysis settings, the two pre-registered values and `runtime.recovery_margin`,
so a frozen analysis can be recognised when the held-out blocks are analysed with the same settings.

## Invariants

- No experiment value has a default here: every value is `core.get(settings, key)`.
- The result depends on the rows and the settings only: not on file order, row order, how rows are split over files,
  the number of worker processes (shards) that wrote them, or the number of threads. Scores are float64, masks bool,
  indices int64.
- Only columns the metrics feature writes are read; no metric is recomputed from a trace.
- A cell whose reference arm has no paired rows is skipped with its reason, never filled in.
- Two blocks overlapping on the same cell are two groups; one block repeating an episode is an error.

## Settings keys read

`profile`, `runtime.ni_margin_accuracy`, `runtime.rank_agreement_threshold`, `runtime.recovery_margin`, and:

`analysis.alpha`, `analysis.bootstrap_resamples`, `analysis.ci_level`, `analysis.baseline`, `analysis.challengers`,
`analysis.ladder`, `analysis.ablations`, `analysis.ablation_reference`, `analysis.containment_reference`,
`analysis.containment_metrics`, `analysis.efficiency_tolerance`, `analysis.adaptive_contrast`,
`analysis.rank_agreement_statistic`, `analysis.rank_agreement_arms`, `analysis.utility_endpoint`,
`analysis.extra_metrics`, `analysis.pair_unit.{mock,llm}`, `analysis.split_by.{mock,llm}`,
`analysis.topology_classes.{mock,llm}.{heterogeneous,homogeneous}` (the topology names differ per platform),
`analysis.blocks.<budget|efficiency|topology|adaptive|adaptive_reference|ladder|ablations|rank|containment>.<mock|llm>` (the
block names are the runner's, `runner/private/blocks.py`).

## Open questions

1. Rank-agreement statistic (spec section 11). Implemented: Spearman rho or Kendall tau on the mean final-round ASR of
   the pre-registered arm pairs, pass iff value >= `runtime.rank_agreement_threshold`. Open: which statistic, which arms
   (with 4 arms Spearman takes 24 distinct values and a threshold of 0.8 allows one swapped adjacent pair), whether the
   ranking is on ASR alone or on a Pareto order of ASR and accuracy, and uncapped runs only or every budget cell. The
   owner sets `analysis.rank_agreement_statistic` and `analysis.rank_agreement_arms`; profiles.toml leaves both unset
   (commented out).
2. Utility endpoint. `accuracy_clean` (attack-free run) or `accuracy_attack`. At the final round ASR and accuracy under
   attack are the same number (1 - ASR, metrics contract open question 1), so with `accuracy_attack` every safety gain is
   a utility gain and `UTILITY_LOSS` cannot occur. D63 names accuracy as the utility endpoint without choosing the run.
   The owner sets `analysis.utility_endpoint`; profiles.toml leaves it unset (commented out).
3. Binary metrics (protected-sink harm, targeted ASR) are tested on episode pairs: (seed, sink) in the mock, conversation
   on the LLM side. Decision D19 says "paired seeds", which with one 0/1 per seed would need a per-seed binarisation.
   Episodes of one seed are not independent; `se_ratio_cluster_vs_naive` shows by how much.
4. Holm family scope. One family per hypothesis, platform, challenger and endpoint (so `BAZT` and `BAZT_upstream` are
   adjusted separately), and attack families, tasks and backbones count as comparisons when `analysis.split_by` lists
   them. D20 names one family per hypothesis (D64 adds the platform), and the code adds the challenger. Open: one family
   across challengers, and whether attack families split or pool.
5. Challengers: `BAZT` only, or also `BAZT_upstream` (decision D86 runs both). The owner sets `analysis.challengers`;
   profiles.toml leaves it unset (commented out).
6. H-topology rule. Each of the three parts uses the hypothesis rule (at least half win, none is a utility loss);
   "absent on Watts-Strogatz" is the class failing that rule. The spec gives no rule for "appears", "absent" or
   "grows with N". The code applies the class rule to appears and absent, and the growth rule of the H-topology row to
   grows with N. The owner confirms or replaces them.
7. H-adaptive contrast. `advantage_shrinks_less_than_c_security`: `delta_ch - 2 * delta_base` (the challenger's advantage
   shrinks by less than the baseline's own loss of security) or `challenger_degrades_less_than_c`: `delta_ch - delta_base`.
   The owner sets `analysis.adaptive_contrast`; profiles.toml leaves it unset (commented out).
8. H-efficiency tolerance: the largest ASR gap to the uncapped baseline that counts as "reaches" (0 means no worse). The
   owner sets `analysis.efficiency_tolerance`; profiles.toml leaves it unset (commented out). The code takes the budget
   total as audit plus intervention fractions; a cell with one budget uncapped has an unbounded total.

## Requests

- Owner (`experiments/profiles.toml`): the commented keys of the block below are not set, and
  `runtime.ni_margin_accuracy`, `runtime.rank_agreement_threshold` and `runtime.recovery_margin` (spec section 10, D77)
  are unset too. The analysis refuses to run until every required key is set; a key may be set to an empty value where
  that is the choice (for example `split_by.mock = []`). The block is the `[analysis]` block of the profiles file.

```toml
[analysis]
alpha = 0.05                               # stats_check.ALPHA; pareto_label labels at this level, any other value is refused
bootstrap_resamples = 1000                 # spec section 7
ci_level = 0.95                            # spec section 7
baseline = "C"                             # H-budget compares with capped C
ladder = ["A", "B", "C"]                   # secondary analysis A > B > C, weakest first
ablations = ["BAZT_no_budget", "BAZT_no_topology", "BAZT_no_graduated"]   # spec 2.6
ablation_reference = "BAZT"
containment_reference = "A"                # spec section 6: the paired no-defense / A run
# challengers = ["BAZT"]                   # OPEN: ["BAZT"] or ["BAZT", "BAZT_upstream"]
# containment_metrics = ["agents_reached_auc"]   # OPEN: gain columns, any of agents_reached_auc, asr_final, protected_sink_harm
# efficiency_tolerance = <float>           # OPEN: ASR gap to uncapped C that still counts as reaching it (0 = no worse)
# adaptive_contrast = "..."                # OPEN: "advantage_shrinks_less_than_c_security" or "challenger_degrades_less_than_c"
# rank_agreement_statistic = "..."         # OPEN: "spearman" or "kendall"
# rank_agreement_arms = [["A", "A"], ["B", "B"], ["C", "C"], ["BAZT", "BAZT"], ["GNN_standin", "GNN"]]   # OPEN: [mock arm, llm arm], at least 3
# utility_endpoint = "..."                 # OPEN: "accuracy_clean" or "accuracy_attack"
# split_by = ...                           # OPEN, see [analysis.split_by]

[analysis.pair_unit]
mock = "seed"                              # spec section 7: mock pairs are seeds
llm = "episode"                            # spec section 7: LLM pairs are conversations

# [analysis.extra_metrics]                 # OPEN: metric = [kind, rows]; kind continuous|binary, rows attacked|attack_free
# protected_sink_harm = ["binary", "attacked"]

# [analysis.split_by]                      # OPEN: fields analysed separately; any of attack, task, backbone
# mock = []
# llm = ["attack"]

# [analysis.topology_classes.mock]         # OPEN: spec 7 names Barabasi-Albert and hub-centred against Watts-Strogatz
# heterogeneous = ["barabasi_albert", "star_hub"]
# homogeneous = ["small_world"]
# [analysis.topology_classes.llm]
# heterogeneous = ["barabasi_albert", "hub_star"]
# homogeneous = ["watts_strogatz"]          # random_directed belongs to neither class unless the owner says so

# [analysis.blocks]                        # OPEN: runner blocks that feed each analysis, per platform
# budget = { mock = ["mock_frontier"], llm = ["llm_capped"] }
# efficiency = { mock = ["mock_frontier"], llm = [] }
# topology = { mock = ["mock_epidemiology"], llm = ["llm_headline"] }
# adaptive = { mock = ["mock_adaptive"], llm = ["llm_adaptive_check"] }
# adaptive_reference = { mock = ["mock_frontier"], llm = ["llm_headline"] }   # non-adaptive runs of the same episodes
# ladder = { mock = ["mock_frontier"], llm = ["llm_headline"] }
# ablations = { mock = ["mock_ablations"], llm = [] }
# rank = { mock = ["mock_frontier"], llm = ["llm_headline"] }
# containment = { mock = ["mock_frontier"], llm = ["llm_capped", "llm_headline"] }   # the LLM A runs only in llm_headline
```

## Known gaps

- Figures of spec section 8 ("statistics and figures") are not produced; the report holds the numbers they would plot
  (frontier cells with intervals, labels per cell, rank table).
- Right-censoring of the recovery times is not handled: `time_to_containment` and `recovery_time_accuracy` can be tested
  as plain continuous `analysis.extra_metrics` (the LLM `nrp` and `recovery_time_accuracy` are formed as above), with
  their censoring ignored.
- The held-out split (mock seeds 31 to 60) is analysed by the same code with `analysis.blocks` pointing at
  `mock_heldout`; the freeze itself (`ANALYSIS_FROZEN`) is enforced by the runner, and the analysis only records the
  settings fingerprint.
- Tested on synthetic rows with planted effects and on the metrics column names; not yet run on rows from the real
  platforms. A 356,000-row synthetic frontier analyses in about 60 s on this machine. Loading dominates (profile at
  119,000 rows: 33 of 46 s, 8 s of it opening 239 files at 35 ms each); the full frontier of millions of rows is not
  measured.
