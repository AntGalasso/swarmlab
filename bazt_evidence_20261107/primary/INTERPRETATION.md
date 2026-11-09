# Preliminary interpretation of the prespecified 10/10 cell

These numbers come from workflow 42381690293 and the frozen (10%,10%) sprint cell. They are descriptive while the full-grid analysis runs. All values below are percentages; differences and intervals are percentage points.

| Topology | ASR C | ASR BAZT | BAZT−C ASR [pointwise 95% CI] | Clean accuracy C | Clean accuracy BAZT | BAZT−C clean accuracy [pointwise 95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| barabasi_albert | 80.46 | 58.35 | -22.11 [-31.80, -12.78] | 75.60 | 76.78 | 1.18 [-0.81, 3.29] |
| chain | 89.88 | 53.83 | -36.05 [-45.56, -26.64] | 65.00 | 67.64 | 2.64 [-5.79, 11.30] |
| small_world | 69.39 | 42.67 | -26.71 [-37.10, -16.45] | 88.01 | 92.45 | 4.44 [1.44, 7.59] |
| star_hub | 80.37 | 80.21 | -0.15 [-1.43, 1.22] | 26.99 | 27.75 | 0.76 [0.19, 1.48] |
| star_leaf | 71.93 | 71.59 | -0.35 [-1.51, 0.85] | 27.87 | 28.29 | 0.42 [0.05, 0.97] |

Thirty seeds are the independent clusters. All 12 sinks and five attack families are averaged within each seed. Clean utility uses each seed's 12 clean sinks once. Bootstrap intervals use 20,000 paired seed resamples and are pointwise: they do not establish full-grid significance, non-inferiority or Pareto labels.

The large ASR reductions on Barabasi–Albert, chain and small-world are encouraging for the BAZT lab stack. Both star topologies show small absolute ASR changes; their clean accuracy is already low because the diameter-scaled horizon leaves Metropolis consensus far from convergence. Report absolute ASR together with clean accuracy and the matched clean-reference degradation; do not label ordinary baseline consensus errors as attack successes caused by the attacker.

The mock gives BAZT and BAZT_upstream a free monitor that C does not receive. These comparisons therefore establish behavior of the evaluated stacks, rather than isolating the allocator/topology mechanism. Insider PI/TA/MA also share closely related parametric dynamics here. A separate LLM task validation is required for broader semantic claims.

BAZT_upstream is a separate adapter and must not inherit BAZT's conclusions. Its means are retained in the JSON/CSV; it loses substantial clean utility on several topologies.

Do not change the already frozen LLM model/task/topology/budget scope after seeing these results. The full budget frontier, adjusted tests, utility non-inferiority and method limitations belong in the final report.

Reproduce: `python extract_ten_ten.py --artifacts-dir ../artifacts_42381690293 --out .` (run from this directory with NumPy installed). Artifact/source/settings hashes and every per-seed value are in ten_ten_summary.json. No raw result rows are changed and no model calls are made.
