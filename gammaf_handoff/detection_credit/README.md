# Can a disabled detector receive detection credit?

An exploratory reanalysis of existing GAMMAF experiment 1 data, prepared on 7 November 2026. GAMMAF is the independent work of Pablo Mateo-Torrejón and Alfonso Sánchez-Macián. The repository revision used here is dated 6 October 2026. The numerical results below describe its released experiment records. [Authors' paper](https://arxiv.org/abs/2604.24477) · [Pinned source and results](https://github.com/pmateo-uc3m/GAMMAF/tree/53be3249ebfa631d8e960c9fb1c931d526654697/experiments-Information-Sciences/experiment1)

## The negative control

On MMLU the no-defence configuration raises zero flags in all 1,137 executed rounds. The same 100 questions are evaluated on four graph topologies, giving 400 debates. Thirty-five debates have an imputed third round. In 32 of those debates the agents reached a correct consensus early and the scoring convention assigns 100% adversarial detection rate (ADR) to that round. In the other three, it assigns zero. The resulting third-round average is 8.00%, with graph-specific values of 7% on the chain, 10% on the random graph, 11% on the star and 4% on the tree. The recorded flags yield zero detected adversaries on every graph. [Scoring code](https://github.com/pmateo-uc3m/GAMMAF/blob/53be3249ebfa631d8e960c9fb1c931d526654697/EvaluationDebateLoop.py#L1187-L1236) · [No-defence records](https://github.com/pmateo-uc3m/GAMMAF/blob/53be3249ebfa631d8e960c9fb1c931d526654697/experiments-Information-Sciences/experiment1/results/summary-exp1-scores/MMLU/no_defense_baseline.json)

| Released task | Published third-round ADR | ADR from last recorded flags |
|---|---:|---:|
| MMLU | 8.00% | 0% |
| MMLU-Pro | 4.75% | 0% |
| CSQA | 2.50% | 0% |
| GSM8K | 0.25% | 0% |
| Memory attack | 0% | 0% |
| Tool attack | 0% | 0% |

The source marks imputed rows explicitly. Its ordinary question-answering rule ties missing-round credit to the correctness of the early consensus. The code has a distinct carry-forward rule for memory attacks and skips this question-answering padding for tool attacks. An endpoint can deliberately reward early task completion. Reporting consensus or correctness alongside actual flags makes the meaning of each quantity explicit.

## Effect on detector comparisons

MMLU was selected because it also featured in the team's earlier LLM pilot and is the first task in the external configuration. All six named detectors are included. Within a debate we compare the released third-round ADR with the fraction of known adversaries flagged at the last executed round. Both scores cover exactly the same questions and topologies. A new counterfactual third round would require different evidence. This replay leaves the executed decisions, model outputs and task accuracy fixed. [External experiment configuration](https://github.com/pmateo-uc3m/GAMMAF/blob/53be3249ebfa631d8e960c9fb1c931d526654697/experiments-Information-Sciences/experiment1/exp1-evaluation.yaml)

| Detector | Released third-round ADR | Last executed ADR |
|---|---:|---:|
| XG-Guard | 87.67% | 99.00% |
| CASPIAN | 81.58% | 75.25% |
| DOMINANT | 73.75% | 68.25% |
| BlindGuard | 65.33% | 63.33% |
| PREM | 65.08% | 71.00% |
| TAM | 44.08% | 40.42% |

Two of the fifteen pairwise mean orderings change. The largest change among these two is for DOMINANT minus PREM, from +8.67 to −2.75 percentage points. The change is −11.42 points. Its approximate simultaneous 95% bootstrap interval is [−20.67, −2.16]. The simultaneous interval for the last-executed difference is [−12.00, +6.50]. The means therefore illustrate sensitivity of a comparison, and the final ordering of this pair remains uncertain. XG-Guard ranks first on both summaries.

The illustrative pair was identified after examining the records. The archive reports all fifteen pairs. For each contrast, the independent resampling unit is the question, with the four graphs averaged within each of the 100 questions. All methods share each question resample. There are 5,000 draws with seed 20261107. Pointwise intervals use bootstrap percentiles. The simultaneous intervals use the 95th percentile of the largest absolute centred bootstrap deviation across all 45 pair/endpoint contrasts in percentage points. Intervals describe sampling sensitivity across this task's questions conditional on the four graphs, the saved model outputs and the chosen scoring conventions.

## Reproduction

Unzip and run the following command in the extracted folder. The replay uses only Python's standard library. It validates the MMLU negative control through the actual flag arrays, checks the padding rules and reproduces the 48 published per-topology third-round ADR values.

```bash
python3 audit.py --output reproduced.json
```

The compressed table contains the necessary numerical projection of 14,400 released round records. These rows include repeated measures. The bundle also includes the expected results and a vector figure. The figure uses distinct, labelled horizontal scales for its two panels. Running `python3 figure.py` redraws it with Matplotlib from `results.json`.

To verify every projected value against the public originals, run these commands from the same folder. The script verifies the Git blob identity and byte count of all 13 source files, then compares every projected row and official per-topology average.

```bash
git clone --filter=blob:none --no-checkout https://github.com/pmateo-uc3m/GAMMAF.git GAMMAF
git -C GAMMAF sparse-checkout set experiments-Information-Sciences/experiment1/results
git -C GAMMAF checkout --detach 53be3249ebfa631d8e960c9fb1c931d526654697
python3 audit.py --checkout GAMMAF --output reproduced.json
```

## Scope for a hackathon report

This case tests whether reported detection credit corresponds to an executed flag. The team's BAZT experiment tests access to a monitor during policy execution. Those are different mechanisms at different stages of evaluation. The practical connection is the need to trace each claimed safety improvement to information the defender received and an action the defender took. This external calculation supplies an example of the latter check. It cannot validate a BAZT-to-LLM transfer, a superior BAZT defence or a defect in the external benchmark's policy implementations.
