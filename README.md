# BAZT: Budgeted Agentic Zero-Trust Enforcement Against Prompt Contagion in LLM Agent Swarms

**Apart Research — Swarm Alignment Hackathon (November 2026)**  
**Authors:** Antonio Galasso, Kunal Sharma, Vittoria Lanzo, Ricardo Fitas  
*Swarm Alignment Research Team*  
**Post-Pilot Configuration SHA-256:** `f8b616b2...5079` | **Pre-Kickoff Tag:** `2026-11-06T20:00:00Z`

---

## Overview

Multi-agent LLM systems can propagate a single malicious instruction from agent to agent along their communication graph. Defending such a swarm at runtime is a resource-allocation problem under uncertainty: judge compute, human review capacity, and intervention budgets are finite, while every defensive intervention removes legitimate collaboration and degrades task consensus.

This repository contains the complete research suite developed during the **Swarm Alignment Hackathon**:
1. **Primary Architecture — BAZT & SwarmLab (`bazt/`, `swarmlab.py`):** A runtime Zero-Trust gateway control plane that mediates inter-agent communications outside the agents' authority, coupled with a 10-policy containment benchmark suite, an asynchronous live-LLM harness integrated with **Adaption Labs** (`qwen3.8-27b`, `deepseek-v4-flash`, `gemma-4-26b-a4b-it`), content-addressed SQLite Common Random Numbers (CRN) caching, and the **`Stop`** budget-aware inspection stopping rule.
2. **Companion Metrological Audit — GAMMAF Causal Audit (`gammaf_handoff/`):** A zero-dependency causal audit and verification suite for the external GAMMAF benchmark (pinned commit `53be3249`, 11,200 debates), isolating post-filter excluded-voter grade leakage (227/9,600 discrepant verdicts), padded-round phantom detection credit (8.00% on disabled controls), and attack-generator target seeding bugs (64/300 gold targets).

---

## Part I: BAZT Architecture & SwarmLab Containment Suite (Primary Focus)

### 1. Core Architectural Mechanisms
BAZT sits at the delivery boundary of a directed communication graph $G=(V,E)$ ($N=12$ agents, scaling up to $N=100$; horizon $T=12$) and enforces five structural principles:

* **Ground-Truth Firewall & Sender-Memory Preservation:** The controller never reads hidden infection states; a paid `inspect(i)` call (cost = 1 audit unit) is its only channel to ground truth. Blocking, sanitizing, or restricting an outgoing edge $(i \to j)$ modifies what recipient $j$ receives without altering the internal state of sender $i$.
* **Evidence Fusion & Provenance Taint:** At round $t$, BAZT fuses the $m=2$ rolling judge score $\rho(i,t)$ with a content-agnostic structural provenance taint $\tau(i,t) \in [0,1]$ (initialized at 1 at ingress nodes, attenuated by $\delta=0.7$ per hop, decaying by $0.8$ per step, and reset to $0$ upon a clean paid inspection):
  $$p'(i,t) = \sigma\!\big(a\,\rho(i,t) - b + w\,\tau(i,t)\big), \qquad a=2.5,\; b=2.0,\; w=1.0$$
* **Finite-Horizon Contagion Reachability:** Expected downstream contagion over remaining horizon $h = T - t$ is estimated via path-decayed reachability on the active subgraph:
  $$H(i) = \sum_{j \in V \setminus \{i\}} (1 - p'_j)\,\gamma^{d(i,j)-1}\,\mathbb{I}\big[1 \le d(i,j) \le \min(h, d_{\max})\big], \qquad \gamma=0.7,\; d_{\max}=3$$
* **Graduated Action Space & Exponential Pacing:** Separate audit and intervention ledgers price five actions by net value $v(a) = b(a) - \lambda(1 - p'_i)u(a)$ and value density $d(a) = v(a)/\mathrm{cost}(a)$:
  1. `inspect` (Cost: 1 audit unit | Utility cost: $0$ | Verdict lifetime: 2 steps, overrides $p'$ to $0.98/0.02$).
  2. `sanitize` (Cost: 1 intervention unit | Strips free-text payload, preserves numeric sensor estimate; $\rho_{\mathrm{san}}=0.1$).
  3. `restrict` (Cost: 2 intervention units | Enforces schema-only output and denies tools for 3 steps; $\rho_{\mathrm{res}}=0.3$).
  4. `block_edge` (Cost: 2 intervention units | Drops delivery along single directed edge $(i \to j)$).
  5. `isolate` (Cost: 3 intervention units | Permanently severs all outgoing edges from node $i$).
  
  To prevent benign early false alarms (*decoys*, such as administrative headers `[DOC-2]`) from exhausting the intervention ledger $B$, candidate actions must clear an exponentially rising density threshold:
  $$d(a) \ge \theta(z) = \theta_0 \exp(\kappa z), \qquad z = \frac{\text{spent}}{B},\quad \theta_0 = 0.150,\; \kappa = 1.5$$
* **Budget-Aware Inspection Stopping (`Stop`) & Terminal Planning Clock:** Once the remaining intervention balance $B_I(t)$ falls below the smallest positive action cost $c_{\min}^+ > 0$, paid inspections on active senders cannot fund any positive-cost intervention. The `Stop` rule suppresses active-sender audits below the price floor unless a zero-cost action (e.g., free redaction $c=0$) is eligible:
  $$M_t^{\mathrm{Stop}} = \begin{cases} M_t & \text{if } B_I(t) \ge c_{\min}^+ \text{ or free redaction } (c=0) \text{ is eligible,} \\ \emptyset & \text{if } B_I(t) < c_{\min}^+ \text{ and all eligible actions have } c > 0. \end{cases}$$
  For live LLM tasks where the receiver reads the final round's incoming delivery before emitting its answer, the planning horizon is set to $h_{\mathrm{LLM}} = \max(T - t + 1, 0)$.

---

### 2. Key Experimental Results (BAZT & SwarmLab)

#### A. Pre-Registered Threshold Calibration (`tune` split: Seeds `9001..9030`, 2,400 episodes)
Thresholds were calibrated on 30 tune seeds by minimizing $J = \mathrm{AUC}_{\mathrm{ever}} + 0.25 \cdot (\mathrm{MAE} / \mathrm{MAE}_{\mathrm{base}})$ and frozen prior to main-split evaluation (`results/tuning_summary.json`):

| Policy Arm | Code Name | Frozen Tuned Parameters | Tune $J \downarrow$ | Tune $\mathrm{AUC}_{\mathrm{ever}}$ | Tune $\mathrm{MAE}$ | Main H3 Dominance Rate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Two-Tier Graduated Defence** | `risk_aware_tiered` | $\tau_{\mathrm{quar}}=2.0, \lambda=0.25, \tau_{\mathrm{san}}=0.35$ | **0.2810** | **0.1609** | **7.03** | **67.9% (19/28, 0 losses — Supported)** |
| Degree-Centrality Quarantine | `centrality_only` | top-$K$ by degree ($K=3$) | 0.3255 | 0.1663 | 9.67 | — |
| Single-Tier Hard Quarantine | `risk_aware` | $\tau_R=0.50, \lambda=0.25$ | 0.3451 | 0.2094 | 8.59 | 41.7% (10/24, 1 loss — Failed) |
| Fixed Early Window | `fixed_early` | $\tau_p=0.50$ | 0.3968 | 0.2510 | 9.30 | — |
| Topology-Blind Score Monitor | `score_only` | $\tau_p=0.50$ | 0.4761 | 0.3425 | 8.61 | — |
| Fixed Mid Window | `fixed_mid` | $\tau_p=0.50$ | 0.6871 | 0.5394 | 10.43 | — |
| Fixed Late Window | `fixed_late` | $\tau_p=0.50$ | 0.8051 | 0.6031 | 15.23 | — |

#### B. Confirmatory Hypothesis Suite (`main` split: Seeds `1..30`, 2,400 episodes) & Gateway Endpoints
* **H1 (Window Timing under Decoys) — `SUPPORTED` (4/4 topologies):** Early intervention beats mid and late windows across all 12 pairwise comparisons (`chain`, `small_world`, `star_hub`, `star_leaf`; Wilcoxon $p_{\mathrm{Holm}} \in [5.16 \times 10^{-6}, 0.00488]$, rank-biserial $r_{\mathrm{rb}} \in [0.68, 1.00]$).
* **H2 (Alert-Driven Decoy Degradation) — `SUPPORTED` (5/8 pairs):** Benign decoy bursts significantly degrade $\mathrm{AUC}_{\mathrm{ever}}$ in 5 of 8 topology–policy pairs ($p_{\mathrm{Holm}} \in [0.0064, 0.0256]$).
* **H3 (Pareto Safety–Utility Dominance, $\mathrm{margin}_{\mathrm{MAE}}=0.05$):**
  * `risk_aware` (Single-Tier Hard Quarantine) **fails H3** ($41.7\%$ dominance, $10/24$ wins, suffering an explicit `UTILITY_LOSS` in `star_hub` vs. `score_only`, $\Delta\mathrm{MAE} = +0.803, p_{\mathrm{Holm}} = 0.0057$).
  * `risk_aware_tiered` (Two-Tier Graduated Defence) **passes H3** ($67.9\%$ dominance, $19/28$ wins: **17 `STRICT_PARETO_WIN`**, 2 `SAFETY_WIN_UTILITY_NONINFERIOR`, and **0 `UTILITY_LOSS` cells**). In `small_world`, it wins 7/7 comparisons (`STRICT_PARETO_WIN`), directly dominating `risk_aware` ($\Delta\mathrm{AUC}_{\mathrm{ever}} = -0.096, p_{\mathrm{Holm}} = 0.00131$; $\Delta\mathrm{MAE} = -5.13, p_{\mathrm{Holm}} = 0.00068$).
* **BAZT Gateway vs. Contextual Zero Trust (Arm C) & Ablations:** Over 24 paired confirmatory cells, BAZT reduces final ASR by **18.40 pp** (95% CI $[+9.72, +28.13]$, paired Wilcoxon $p = 0.0024$) and increases consensus accuracy by **10.56 pp** (95% CI $[+0.69, +11.46]$, non-inferiority $p = 0.0088$). Removing graduated actions (`bazt_binary`: allow or isolate only) reaches the same $16.67\%$ ASR floor but costs **33.33 pp**, **12.50 pp**, and **4.17 pp** of accuracy on Barabási–Albert, Watts–Strogatz, and Star topologies.

#### C. Full-Grid Live LLM Sweep on Adaption Labs (2,400 episodes, `qwen3.8-27b`)
Executed across all 30 main seeds, 4 topologies, 2 regimes (`clean`, `decoy`), and 10 policies (`results/episodes_main.jsonl`):

| Topology (`decoy` regime) | `no_defense` MAE | `score_only` MAE | `risk_aware` MAE (Collat / DecoyW) | `risk_aware_tiered` MAE (Collat / DecoyW) | Hub $\mathrm{AUC}_{\mathrm{ever}}$ (`dRel`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`chain`** | 6.39 | 6.51 | 6.88 (3.00 / 0.27) | **6.09** (**1.20** / **0.12**) | 0.003 (0.00) |
| **`small_world`** | 6.72 | 6.82 | 7.26 (2.97 / 0.31) | **6.65** (2.97 / **0.27**) | 0.003 (0.00) |
| **`star_hub`** | 7.98 | 8.03 | 8.15 (3.00 / 0.26) | **7.92** (**2.67** / **0.24**) | **0.000 (+16% / +19%)** |
| **`star_leaf`** | 6.74 | 6.78 | 6.83 (3.00 / 0.23) | **6.60** (**2.70** / **0.19**) | **0.000 (+2%)** |

* **Live Outbreak Containment:** In `star_hub` Seed 6, where unmitigated infection erupts at $t=5$ ($\mathrm{AUC}_{\mathrm{ever}} = 0.0556$, final infection rate $8.33\%$), all topology-blind policies fail, whereas `risk_aware_tiered` isolates patient zero at $t=1$ ($\mathrm{AUC}_{\mathrm{ever}} = 0.000$), spending only 1 quarantine unit in `clean` vs. 3 for `risk_aware`.
* **Consensus Preservation under Decoys:** Because natural contagion is sparse on `qwen3.8-27b`, hard quarantine (`risk_aware`) mostly isolates healthy nodes ($\mathrm{collat} \approx 3.00$), degrading MAE below `no_defense` in all topologies. By routing intermediate alarms to tier-1 sanitization ($\mathrm{san} \in [2.8, 4.0]$), `risk_aware_tiered` achieves the **lowest MAE across all four topologies**.

#### D. Multi-Conversation Live Gateway Pilot (`deepseek-v4-flash` + `qwen3.8-27b` judge)
On Barabási–Albert graphs ($N=12$, 4 sequential conversations $\times$ 3 rounds per episode):
* **Undefended (Arm A):** Reaches **79.17% final ASR** ($29.17\%$ accuracy).
* **Capped (10% Audit, 10% Intervention):** BAZT lowers mean ASR from **70.83% (Arm C) to 52.08%** ($-18.8\text{ pp}$) and raises accuracy from **20.83% to 34.38%** ($+13.5\text{ pp}$) with zero budget violations.
* **Uncapped:** BAZT reaches the **16.67% theoretical ASR floor** ($58.33\%$ accuracy) spending **13.5 audit units** vs. **192.0** for stateless judge Arm B.

#### E. Prospective Evaluation of the `Stop` Rule (4,320 Simulator Pairs & 30 Live `gemma-4-26b-a4b-it` Worlds)
* **Numerical Consensus Simulator (60 seeds, 4,320 clean/attacked pairs, 10% cap):** `Stop` reduces purchased inspections from $20.31$ to $17.89$ (**-11.91%**, 95% CI $[-14.70\%, -9.28\%]$) and total charged budget units from $37.98$ to $35.56$ (**-6.37%**, 95% CI $[-7.80\%, -5.01\%]$), with byte-identical excess error ($18.38\%$) and clean accuracy ($63.89\%$), capturing $91.5\%$ of the savings of oracle lookahead `Selective` while running $3.1\times$ faster ($0.230\text{ s}$ vs. $0.727\text{ s}$).
* **Live Distributed Record Lookup (30 held-out worlds, 12 `gemma-4-26b-a4b-it` actors + `qwen3.8-27b` monitor):** Eliminates all $169$ ($1\%$ cap) and $379$ ($5\%$ cap) below-floor sender inspections, cutting total inspections by **16.52%** and **36.65%**, judge token cost by **12.09%** and **30.10%**, and total token cost by **0.203%** and **0.526%**, with **0/30 final-answer disagreements** ($29/30$ and $28/30$ correct).

---

## Part II: Companion Causal Audit of the GAMMAF Benchmark (`gammaf_handoff/`)

During our empirical profiling of public graph-defense traces from **GAMMAF** (*Graph-Based Anomaly Monitoring Benchmarking in LLM Multi-Agent Systems*, pinned commit `53be3249`), we discovered that excluded insider agents can still flip the benchmark's published correctness score. We isolated three metrological defects across the **11,200 archived debates** ($9,600$ detector-enabled across `BlindGuard`, `CASPIAN`, `DOMINANT`, `PREM`, `TAM`, `XG-Guard` + $1,600$ disabled controls on `MMLU`, `MMLU-Pro`, `CSQA`, `GSM8K`):

1. **Defect 1 — Post-Filter Vote Leakage (`227/9,600` Discrepant Verdicts, `2.36%`):**
   While the answer delivered to the user $a_{\mathrm{del}} = \pi(v_U)$ aggregates only unflagged eligible voters $U = \{i : f_i = 0\}$, GAMMAF's released grading function $g_{\mathrm{release}} = \mathrm{check}(\pi(v), y)$ tallies all 10 terminal votes $v$ (including flagged insiders $\bar{U} = \{i : f_i = 1\}$).
   * **156 False Credits:** The delivered answer is wrong ($g_{\mathrm{del}} = 0$), but excluded insiders' votes make the all-voter plurality match $y$ ($g_{\mathrm{release}} = 1$). `PREM` receives 62 false credits vs. 4 missed credits, inflating its accuracy by **+3.625 pp** ($1,363/1,600$ released vs. $1,305/1,600$ delivered).
   * **71 Missed Credits:** The delivered answer is right ($g_{\mathrm{del}} = 1$, e.g., `MMLU-Pro` ID `6108` under `XG-Guard`), but excluded insiders swing the all-voter plurality to a wrong option ($g_{\mathrm{release}} = 0$).
   * **Single-Vote Counterfactual Sensitivity:** Editing a single already-excluded insider's terminal vote flips $g_{\mathrm{release}}$ in **121 debates** (89 flagged in the prior round) while changing **0 delivered answers** across **33,025 valid substitutions**. Restricting to canonical answers with unique pluralities in both populations retains **41/4,301** discrepancies ($36/4,177$ on the 4-letter alphabet).
   * **Detector Ranking Impact:** Switching $g_{\mathrm{release}} \to g_{\mathrm{del}}$ reverses the top-ranked detector in **2/16 task–topology cells** (`CSQA tree`: `TAM` $91\to 88$ vs. `XG-Guard` $88\to 89$; and `MMLU random`), and creates a development tie on `MMLU` ($328/400$) where alphabetical selection of `TAM` reduces reserved-task delivered accuracy by **1.75 pp** (95% CI $[0.33, 3.25]$).
2. **Defect 2 — Phantom Detection Credit from Round Padding (`8.00%` on Disabled Controls):**
   In detector-disabled controls on `MMLU`, zero flags were raised across $1,137$ executed rounds, yet post-consensus round imputation awarded an **8.00% detection credit** at round 3.
3. **Defect 3 — Attack Generator Target-Seeding Bug (`64/300` Gold Targets, `21.3%`):**
   Comparing integer indices ($0..3$) against character gold labels (`'A'..'D'`) and slicing only the first 4 options caused the attack generator to seed the **correct gold answer as the adversarial target in 64/300 multiple-choice questions** ($28/100$ `MMLU`, $21/100$ `CSQA`, $15/100$ `MMLU-Pro`). Our patch normalizes option letters, excludes $y_{\mathrm{gold}}$, and passes all 780 seed checks.

---

## Repository Structure

```text
swarmlab-bazt/
├── bazt/                         # BAZT gateway control plane package (ledger, optimizer, taint, reachability)
├── bazt_lab/                     # Multi-block experiment runner & GNN/judge calibration modules
├── gammaf_handoff/               # Standalone zero-dependency GAMMAF causal audit & reproduction runner
│   ├── run.py                    # Single-command verifier (31 public files, 11,200 debates, 33,025 edits)
│   └── README.md                 # Handoff instructions & public-source rebuild guide
├── results/                      # Committed experimental artifacts & statistical outputs
│   ├── tuned_params.json         # Frozen thresholds calibrated on tune split (9001..9030)
│   ├── tuning_summary.json       # Full grid search metrics for J = AUC_ever + 0.25 * (MAE / MAE_base)
│   ├── stats_report_main.json    # Wilcoxon + Holm-Bonferroni + 10k bootstrap CIs for H1, H2, H3
│   ├── episodes_main.jsonl       # Per-episode traces for the 2,400-episode live qwen3.8-27b sweep
│   ├── per_seed_main.csv         # Per-seed metrics across topologies, regimes, and policies
│   └── figures/                  # Vector PDF plots (fig1_pareto..fig5_constraint_evasion)
├── scripts/
│   ├── e2e_mock.py               # Fast end-to-end interrupt/resume & pipeline verification test
│   └── e2e_common.py             # Test fixtures and runtime parameter definitions
├── swarmlab.py                   # 10-policy containment simulator & live-LLM evaluation engine
├── llm_client.py                 # Async OpenRouter / Adaption Labs client + WAL SQLite CRN cache
├── stats_check.py                # Non-parametric hypothesis testing script (H1-H3)
├── make_figures.py               # Publication figure generator
└── requirements.txt              # Python dependencies
