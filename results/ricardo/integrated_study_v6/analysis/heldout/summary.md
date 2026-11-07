# Heldout evaluation

120 policy episodes; 6 independent seeds, paired across two topologies.

| Regime | Policy | Mean AUC | Mean MAE | Mean enforcement spend |
|---|---|---:|---:|---:|
| clean | no_defense | 0.6788 | 16.3995 | 0.00 |
| clean | score_only | 0.6696 | 6.7108 | 12.00 |
| clean | centrality_only | 0.1406 | 8.8280 | 12.00 |
| clean | bazt | 0.1215 | 6.4808 | 8.42 |
| clean | bazt_no_pacing | 0.1042 | 6.0175 | 7.75 |
| decoy | no_defense | 0.6678 | 16.1339 | 0.00 |
| decoy | score_only | 0.6574 | 8.5283 | 12.00 |
| decoy | centrality_only | 0.1470 | 9.0254 | 12.00 |
| decoy | bazt | 0.3692 | 14.0546 | 10.08 |
| decoy | bazt_no_pacing | 0.3455 | 13.0012 | 11.00 |

H2 decoy penalty: -0.0122; 95% CI [-0.029513888888888912, 0.0023148148148148437]; one-sided p=0.87500.

H3 uses a 12-test Holm family for safety and utility in both regimes. See statistics.json for every contrast and interval.
These endpoints concern simulator-supplied numeric estimates, a marker-based monitor, and a paid simulator oracle. They are not general LLM safety measures.
