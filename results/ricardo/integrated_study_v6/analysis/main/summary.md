# Main evaluation

240 policy episodes; 12 independent seeds, paired across two topologies.

| Regime | Policy | Mean AUC | Mean MAE | Mean enforcement spend |
|---|---|---:|---:|---:|
| clean | no_defense | 0.6635 | 16.6515 | 0.00 |
| clean | score_only | 0.6435 | 6.2616 | 12.00 |
| clean | centrality_only | 0.1701 | 9.9122 | 12.00 |
| clean | bazt | 0.0929 | 5.1607 | 8.25 |
| clean | bazt_no_pacing | 0.0929 | 5.2084 | 7.50 |
| decoy | no_defense | 0.6667 | 16.8400 | 0.00 |
| decoy | score_only | 0.6603 | 9.0937 | 12.00 |
| decoy | centrality_only | 0.1701 | 9.8540 | 12.00 |
| decoy | bazt | 0.3203 | 12.2914 | 10.21 |
| decoy | bazt_no_pacing | 0.2593 | 11.3406 | 11.88 |

H2 decoy penalty: 0.0168; 95% CI [-0.0014467592592592542, 0.037037037037037014]; one-sided p=0.09375.

H3 uses a 12-test Holm family for safety and utility in both regimes. See statistics.json for every contrast and interval.
These endpoints concern simulator-supplied numeric estimates, a marker-based monitor, and a paid simulator oracle. They are not general LLM safety measures.
