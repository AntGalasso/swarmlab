# Completed primary mock — descriptive results

Run 42381690293 completed successfully in all five topology jobs. The strict merge and an independent raw scan validated exactly 2,127,600/2,127,600 records, thirty independent seeds and twelve sinks per seed. There were no duplicate/missing coordinates, uncommitted files or enforcement-budget violations.

The sprint budget (10% audit, 10% intervention) was fixed before viewing outcomes. Equal-family mean final incorrectness, C versus BAZT_lab:

| Topology | C | BAZT_lab | Difference, percentage points |
|---|---:|---:|---:|
| Barabasi-Albert | 80.46% | 58.35% | -22.11 |
| Chain | 89.88% | 53.83% | -36.05 |
| Small-world | 69.39% | 42.67% | -26.71 |
| Star-hub | 80.36% | 80.21% | -0.15 |
| Star-leaf | 71.93% | 71.59% | -0.35 |

The descriptive paired-seed bootstrap intervals exclude zero for final incorrectness in the first three topologies. These are pointwise intervals, not a familywise non-inferiority guarantee. In particular, chain's clean-utility interval permits a 5.79 percentage-point loss despite its positive mean utility difference. The upstream BAZT implementation also has material clean-utility losses in several conditions. Do not transfer the core implementation's positive result to upstream automatically.

The full frozen cross-grid analysis has now completed. The global H-budget criterion is NOT supported: BAZT has 105/1,200 safety-win/utility-noninferior labels (8.75%, below the required 50%), 169 safety gains without established utility non-inferiority, 926 no-significant-difference labels and no UTILITY_LOSS labels. BAZT_upstream has no joint-win labels, 255 UTILITY_LOSS labels, 28 safety gains without utility non-inferiority, and 917 no-significant-difference labels. There are no strict Pareto-win labels. No global superiority claim is warranted.

The descriptive efficiency analysis reaches the final incorrectness of uncapped C at finite budgets in 25/25 core BAZT frontiers and 20/25 upstream frontiers. These are point-estimate frontier results with bootstrap uncertainty, not 25 independent hypothesis confirmations. The A/B/C ladder is not supported in any of its 25 contexts. Topology/adaptive/ablation/rank-agreement hypotheses were not evaluated in this run.

The predeclared comparison labels use Holm for zero-difference tests and pointwise bootstrap intervals for non-inferiority. Do not describe that as simultaneous familywise non-inferiority.

The mock gives BAZT free parametric monitor evidence that C does not receive. This is evidence about the evaluated stacks, not identification of the allocator's isolated causal effect. Final incorrectness includes ordinary consensus error; star graphs have a low clean floor under the short horizon. PI/TA/MA also share similar numerical abstractions. The separately frozen LLM pilot remains essential and has not yet produced benchmark outcomes.

See ten_ten_summary.json/CSV, extract_ten_ten.py, INTERPRETATION.md, the independent audit and merge_validation.json for calculations and provenance. All five original Actions artifacts remain the authoritative raw records.
