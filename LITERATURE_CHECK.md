# LITERATURE_CHECK.md: what the cited sources actually support

Checked on 2026-11-04 against the primary texts (not summaries) for the BAZT experiment design.
Tags: **[S]** the source states it, **[I]** our inference. Page and section locators refer to the version
named in each entry. Use this file when writing related work and when choosing what the paper may claim.

## 1. Citation verdicts

| # | reference | verdict | notes |
|---|---|---|---|
| 1 | Rose et al. 2020, NIST SP 800-207 | correct | doi:10.6028/NIST.SP.800-207, 59 pp. |
| 2 | Chandramouli & Butcher 2023, NIST SP 800-207A | correct, title differs | PDF cover says "...in Multi-**Location** Environments"; the CSRC page says "Multi-Cloud". Cite the PDF title. |
| 3 | Zhang et al. 2025, ASB, ICLR 2025 | correct | arXiv 2410.02644; no conference DOI. |
| 4 | Debenedetti et al. 2026, CaMeL, SaTML 2026 | correct | doi:10.1109/SaTML68715.2026.00040, pp. 587–618. Numbers differ between arXiv versions: cite the version you quote. |
| 5 | Lee & Tiwari 2024, Prompt Infection | correct | arXiv 2410.07283 v1, preprint only, no peer-reviewed venue found. |
| 6 | Greenblatt et al. 2024, AI Control, ICML | correct | PMLR 235:16295–16336; arXiv 2312.06942 v5 is longer than the ICML version. |
| 7 | Pastor-Satorras & Vespignani 2002 | correct | PRE 65, 036104. |
| 8 | Wang et al. 2026, VLDB Journal 35(6):53 | correct (Crossref rechecked) | doi:10.1007/s00778-026-01003-4, online 3 Sep 2026; extends PVLDB 17(10) 2024 (arXiv 2405.12871). "26 pages" not verified. |

## 2. Corrections to the BAZT concept document

1. **ASB is cited for the wrong claim.** ASB supports "agents introduce attack surfaces through prompts, tools,
   memory" [S]. It evaluates no capability or information-flow control; its defences are prompt-level
   (delimiters, sandwich, paraphrase, rewriting) and detection (perplexity, LLM-based) [S, Table 2, §4.4].
   Cite CaMeL alone for capability and information-flow control.
2. **CaMeL's scope is narrower than the sentence implies.** It assumes a trusted user prompt and uncompromised
   memory (§3), is single-agent (one P-LLM/Q-LLM pair), and constrains tool calls, not everything the agent
   does [S]. "Provable security" in the abstract has no formal proof; formal verification is future work (§10) [S].
3. **"Cloud-native ZTA supports fine-grained, service-level enforcement"** comes from 800-207A, not 800-207.
4. **Pastor-Satorras & Vespignani do not support topology targeting on our graphs.** Targeted (highest-degree)
   immunization beats uniform only on heavy-tailed networks at large N. On Watts-Strogatz graphs the two perform
   the same: "targeted strategies do not have a particular efficiency in systems with limited heterogeneity"
   [S, §IV.B, Fig. 1a]. At N ≤ 36 there is no phase transition, so threshold arguments do not carry over [I].
5. **Prompt Infection gives no grounding for topology claims.** Its "society" uses random pairwise dialogues
   (10–50 agents), and spread depended on manipulating memory importance; without it the infection died out
   [S, §5.2, Fig. 6b]. It studies no fixed graph, budget, or adaptive defender [I].
6. **Wang et al. is methodological only.** Independent Cascade, one-shot node blocking before diffusion, known
   and unblockable seeds, unit costs, no false-positive cost [S, §2.2]. The objective is monotone but not
   submodular (Thm 2), so a greedy pick carries no approximation guarantee [S].

## 3. What each foundation contributes to the design

**Zero Trust (800-207, 800-207A).** ZT is "not a single architecture but a set of guiding principles" [S, §1 p1].
Components: Policy Engine (decides grant, deny, revoke via a trust algorithm), Policy Administrator, Policy
Enforcement Point on every path [S, §3, Fig. 2]. The trust algorithm is criteria-based (all rules must pass) or
score-based (weighted confidence against a per-resource threshold), and singular (each request alone) or
contextual (uses recent history; NIST prefers contextual) [S, §3.3.1 pp19–20]. Decisions are made per request or
per session and re-evaluated continually [S, tenets 3 and 6, p7]. Options beyond grant and deny exist only as a
list: reduce privileges, step-up authentication, pause, deny all connections of a subverted asset [S, pp19–20, p7].
**NIST defines no verification budget or cost model**; cost appears only qualitatively ("cost-efficiency", tenet 6)
[S]. ZT limits lateral movement by micro-segmentation, with no graph reasoning (no centrality, no propagation
analysis) [I from absence]. 800-207A adds sidecar PEPs in a service mesh and per-request, identity-tier
service-to-service authorization [S, §2.1, §3]. **The decision is deterministic given its inputs; uncertainty sits
in the inputs** ("minimize uncertainty", which "cannot be eliminated", §2 p4) [S].

**AI Control (Greenblatt et al.).** Scarce resource: a 2% human **audit** budget [S, §2.2]. Safety = probability
that no backdoor is ever submitted against the attacker's best response; usefulness = correctness with the honest
model, rescaled between the trusted-only and untrusted-only extremes, measured without the attacker [S, §2, §3,
App. J]. Protocols are swept over thresholds and compared as safety-usefulness curves [S, Fig. 2]. Routing is
deterministic given monitor scores; audit thresholds are set by false-positive rate on honest outputs [S, §3.2].

**ASB and CaMeL.** Reusable metrics: attack success rate, benign utility (ASB's PNA), utility under attack,
NRP = PNA × (1 − ASR) [S, ASB Eq. 18–20]. CaMeL excludes policy denials from its utility figure while policies
fired on 33.9–53.2% of solved benign tasks [S, Table 8]; we must not repeat that: a blocked benign message is a
utility cost.

## 4. Related work the paper must cite and distinguish

| work | what it does | threat to BAZT novelty |
|---|---|---|
| Roy et al. 2026, **Agentic-ZTA**, arXiv 2610.05782 | LLM agents **act as** the policy engine for human and device access in an IoBT testbed; binary decision with a confidence threshold per zone | low: agents enforcing ZT vs BAZT, which enforces ZT over agents |
| Le et al. 2026, Cybersecurity 9:191, doi:10.1186/s42400-026-00628-w | epidemic model of prompt-injection spread in agent swarms (SEIQ coupled to SIS), epidemic threshold, **closed-form budget allocation** | **high**; full text not yet read (paywalled), read before submission |
| G-Safeguard, arXiv 2502.11127 | graph neural network over the message graph detects attacks, then topological intervention | moderate: topology-aware, but no budget or ZT framing |
| Huang et al. 2025, arXiv 2508.12259 | identity framework for agents, trust-adaptive runtime that tightens containment as trust falls | moderate on graduated response only |
| Safin et al. 2026, arXiv 2609.17648 | signed task tokens at agent boundaries, multi-hop propagation in a 4-agent pipeline | low |
| Bhatt et al. 2025, Ctrl-Z, arXiv 2504.10374 | control protocols for multi-step agents, 1% audit budget, resampling | framing reference |
| Griffin et al., arXiv 2409.07985 | control as a game; some optimal protocols are randomized | framing reference |
| Cohen, Havlin & ben-Avraham 2003, PRL 91, 247901 | acquaintance immunization | baseline reference |

## 5. Claims the paper must not make

- That NIST ZTA has a budget, a topology model, or a formal escalation ladder.
- That Pastor-Satorras & Vespignani support targeting on small or degree-homogeneous graphs.
- That greedy budgeted blocking has an approximation guarantee.
- That our mock evaluation is an AI Control evaluation: there is no adaptive red team. Call it "control-inspired"
  unless an attacker that knows BAZT and best-responds is added.
- That mock numbers are comparable to ASB or AgentDojo numbers.
- First epidemic model of prompt infection (Agent Smith, Prompt Infection, Le et al.), first budgeted MAS defence
  (BRA-Audit, MESA, Le et al.), first topology-aware defence (G-Safeguard, BlindGuard, MAS-Shield), first
  security-utility trade-off (Multi-Agent Security Tax, AI Control), first graduated response (A-Trust, MAS-Shield,
  AI Control).
- "Zero trust fails" in general: the baseline is *per-message-deny* ZT, and must be named that way.
- A novel centrality: path-decayed reachability is close to Katz centrality.
- Coverage of shared-tool contagion (Le et al.'s core channel) unless the simulator models a tool layer.

## 6. How many "types" of ZTA exist

There is no single count; each framework counts along its own axis (second literature pass, 2026-11-04).

| axis | types | source |
|---|---|---|
| approach (policy driver) | 3: enhanced identity governance, micro-segmentation, network infrastructure / SDP ("a full ZT solution will include elements of all three"); SASE appears as a 4th in practice | 800-207 §3.1, pp11–12; SP 1800-35 §3.4 |
| deployment model (where the PEP sits) | 4: device agent/gateway, enclave gateway, resource portal, application sandboxing | 800-207 §3.2.1–3.2.4, pp13–16 |
| trust algorithm | criteria- vs score-based; singular vs contextual; trust quantification by probabilistic, reputation, fuzzy or ML models | 800-207 §3.3.1; Nam et al. 2026, IEEE Access, doi:10.1109/ACCESS.2026.3712376, §IV |
| scope (pillars) | CISA ZTMM v2.0: 5 pillars + 3 cross-cutting capabilities; DoD: 7 pillars; Forrester ZTX: 7 pillars | CISA ZTMM v2.0 §5; DoD and Forrester via secondary sources |
| maturity | CISA: Traditional, Initial, Advanced, Optimal | CISA ZTMM v2.0 |
| adaptivity | static rules vs adaptive / risk-based (Gartner CARTA; Bayesian trust with dynamic games) | Ge & Zhu, arXiv 2312.02882 §1.3.2, §1.6 |

Cost appears in every ZT source only qualitatively; Buck et al. 2021 (Computers & Security,
doi:10.1016/j.cose.2021.102436) note that "economic analyses … have been neglected".

**Zero trust for AI agents** splits into "ZT applied to agents" and "agents as ZT enforcers" (Agentic-ZTA). Our
synthesis of 2024–2026 work gives six families [I]:

| family | what is enforced | examples |
|---|---|---|
| A. agent identity, delegation, authorization | who may contact whom (tokens, sessions) | Entra Agent ID, Google Agent Identity, NCCoE agent identity concept paper (Feb 2026), CSA Agentic AI IAM, Huang 2508.12259, SAGA 2504.21034 |
| B. least privilege over tools | per tool call | Progent 2504.11703, MiniScope 2512.11147, AgentSentinel 2509.07764, Irreversibility Budget 2609.00275 (value-at-risk budget per principal, not per message, no topology) |
| C. information flow inside one agent | data and control flow | CaMeL, FIDES 2505.23643, f-secure 2409.19091, IsolateGPT, ACE 2504.20984 |
| **D. inter-agent communication and graph defences (BAZT's family)** | messages, edges, agents | G-Safeguard, BlindGuard 2508.08127, INFA-Guard 2601.14667, SAIGuard 2606.12474, SentinelAgent 2505.24201, MAS-Shield 2511.22924, MESA 2606.30602, BRA-Audit 2608.14668 |
| E. adaptive trust scoring | agent and message trust | A-Trust 2506.02546, SentinelNet (WWW'26), CSA Agentic Trust Framework |
| F. agents as ZT enforcers | human and device access | Roy et al. Agentic-ZTA 2610.05782 |

Adjacent: AI Control (Ctrl-Z 2504.10374; Hua et al. 2507.15886 with an average-case budget on monitors and
interventions). SoK 2609.00595 (197 works) lists interventions (block, rewrite, reroute, quarantine, revoke) but
never ranks or costs them.

## 7. Novelty: closest prior work (feature matrix)

Columns: (a) budget, (b) topology-aware, (c) graduated actions, (d) utility-loss accounting, (e) comparison to
zero trust, (f) real LLM agents. Y = yes, P = partial, N = no.

| work | a | b | c | d | e | f |
|---|---|---|---|---|---|---|
| **BRA-Audit**, Wang, Lin, Lou, Li, arXiv 2608.14668 (Aug 2026) | Y, audits per round | Y, dependency graph | N, block and roll back | Y | P | Y |
| **MESA**, Li, Domico, Gregory, McDaniel, arXiv 2606.30602 (Jun 2026) | Y, monitor top 10% edges | Y, 6 graph metrics + probes | N | ? | N | Y |
| **MAS-Shield**, Wang et al., arXiv 2511.22924, EMNLP Findings 2026 | P | Y, high-influence agents | P, tiered *auditing* | Y, latency | N | Y |
| Le et al. 2026, Cybersecurity 9:191 | Y, design-time split between agent and tool layers | Y, spectral | N | N | P, argued only | N, simulation; 0% ASR in their own LLM red-teaming |
| G-Safeguard (ACL 2025) / BlindGuard | N | Y | N, edge cut | P | N | Y |
| XG-Guard 2512.18733 | P, top-3 agents | Y | N | ? | N | Y |
| A-Trust 2506.02546 (NeurIPS 2026) | N | N | P, message vs agent | Y | N | Y |
| Multi-Agent Security Tax 2502.19145 (AAAI 2025) | N | N | N | Y | N | Y |
| AI Control (ICML 2024), Ctrl-Z | Y | N | Y | Y | N | Y |
| NetShield-style immunization | Y | Y | N | N | N | N |

**Defensible claim [I]:** a runtime, per-step controller that chooses *among* actions with different cost and
duration (one-step sanitize, one-message deny, permanent isolate) under an explicit enforcement budget, ranking by
monitor evidence times path-decayed reach minus a utility cost, evaluated head-to-head against per-message ZT at
matched budget. Each ingredient exists; the combination and the ZT comparison do not, in what we found. A final
arXiv sweep is still needed before submission.

**Reviewer-expected baselines, implementable in the mock in about a day:** BRA-Audit greedy placement
(weight = time since last audit × descendants, budget ρ·|candidates| per round, binary block); G-Safeguard-style
detect-and-prune (top-k by monitor score, cut outgoing edges, same monitor instead of a trained GNN, said
explicitly); static topological immunization (degree, eigenvector, NetShield); optionally an A-Trust-style policy.

## 8. Auditor operating points (grounds the mock auditor)

| setting | TPR | FPR (benign) | FPR (suspicious-looking benign) | anchors |
|---|---|---|---|---|
| strong-model audit | 0.99 | 0.005 | 0.03 [I] | PromptArmor GPT-4o/4.1 on AgentDojo (FNR ≤ 0.23%, FPR 0.07–0.56%, arXiv 2507.15219 Table 1); DataSentinel in-distribution FPR 0.00–0.01 (arXiv 2504.11358); Ctrl-Z assumes infallible audit |
| good classifier | 0.95 | 0.02 | 0.12 | PromptShield-8B 94.8% TPR at 1% FPR (arXiv 2501.15145 Table 4); AI Control's 2% audit FPR; best over-defence on NotInject 12.4–13.3% (arXiv 2410.22770 Table 1) |
| weak LLM judge | 0.80 | 0.20 | 0.45 | DeBERTa on AgentDojo 0.78/0.28; PromptArmor-GPT-3.5 0.84/0.11; SHADE-Arena 80% TPR needs > 20% FPR (arXiv 2506.15740 §4.1); ProtectAI v2 on NotInject ≈ 43% |

Under adaptive attack, detection collapses: 61% to 1–10% (Zhan et al., arXiv 2503.00061 Table 2); > 90% ASR
against ProtectAI, PromptGuard, Model Armor (Nasr, Carlini et al., arXiv 2510.09023 §5.3). No source measures
auditing of a single inter-agent message; all anchors are proxies. Vendor numbers (Prompt Guard 2, PINT) are
in-distribution.

## 9. Design pilot: does topology targeting show at N = 12?

Mock dynamics, design seeds 9201–9260 (disjoint from every graded split), static isolation of a fraction `g` of
nodes at t = 1 chosen at random or by highest degree. Value = (AUC_random − AUC_degree) / AUC_none.
**Pipeline check used to choose the design, not a finding.** Script:
`scratchpad/epi_immunize.py` (to be moved into the repo as a design tool).

| graph | N | κ = ⟨k²⟩/⟨k⟩ | g = 0.1 | g = 0.2 |
|---|---|---|---|---|
| small-world | 12 | 4.13 | −0.09 | −0.03 |
| small-world | 100 | 4.22 | +0.30 | +0.26 |
| scale-free (BA, m = 2) | 12 | 4.46 | +0.12 | +0.28 |
| scale-free | 50 | 6.63 | +0.56 | +0.49 |
| scale-free | 100 | 7.67 | +0.58 | +0.55 |
| scale-free | 200 | 8.64 | +0.65 | +0.43 |

## 10. Agent history, persistence of compromise, recovery (evidence only)

**Term check.** "Budgeted zero trust", "budget-aware zero trust", "BZTA", "BAZT", "budgeted agentic zero trust": no
named concept found (six web queries, not an exhaustive Scholar sweep). Hits concern money budgets only.

**New closest prior art: DynaTrust**, Li, Hu, Zhang, Quan, Yu, Wang, arXiv 2603.15661 (Mar 2026), verified: a dynamic
trust graph, trust updated from each agent's past behaviour, compromised agents isolated by restructuring the graph
while tasks stay connected; main baseline AgentShield (+41.7% defence success). A reader reported a per-agent Beta
trust with asymmetric penalty (Eq. 2–3) and a zero-trust baseline with FPR 15.3% vs 2.2% (Table 1): **not yet
verified against the full text.**

| model | how history enters | recovery | source |
|---|---|---|---|
| NIST SP 800-207 | "historical subject behavior patterns" are a trust-algorithm input; contextual algorithms use recent history and are preferred | no rule given | §3.3 p17–18, §3.3.1 p19–20 [S] |
| Beta reputation | counts of good and bad evidence, with a forgetting factor λ (λ = 1 forgets nothing) | yes if λ < 1 | Jøsang & Ismail 2002, §2.3, §2.6 [S] |
| Bayesian trust in ZT | posterior becomes the next prior | implied | Ge & Zhu, arXiv 2312.02882 §1.3.3 [S] |
| A-Trust | violation rate over a recent window, removal at > 80% of the last 20 queries | none, removal permanent | arXiv 2506.02546 §4.3 [S] |
| SentinelNet | per-round scores, cumulative blacklist | none | arXiv 2510.16219 §4.3.2 [S] |
| Huang et al. | weighted trust with exponential decay; containment inversely proportional to trust | named, unspecified | arXiv 2508.12259 §IV-E, §VI [S] |
| CSA Agentic Trust Framework | autonomy levels; critical incident demotes to the lowest level | re-earn through time and zero-incident gates | CSA blog 2026-02-02 and its maturity model (metrics "illustrative") [S] |
| Microsoft Entra ID Protection | confirmed compromise blocks and revokes tokens | only after credential rotation | learn.microsoft.com, "Remediate risks and unblock users" [S] |
| DynaTrust | per-agent trust from past behaviour | replace with a clean-memory replica | arXiv 2603.15661 §4.5 (per reader, unverified) |

**Persistence of compromise.** Injected instructions persist through stored state: ChatGPT memory (Rehberger,
"SpAIware", 2024), RAG/memory poisoning (AgentPoison, arXiv 2407.12784: > 80% attack success at < 0.1% poison rate;
MINJA, arXiv 2503.03704: 98.2% injection success), and Agent Smith's image album, where an agent recovers only when
the adversarial item is pushed out (arXiv 2402.08567). Counter-evidence: in MINJA, memory poisoned by an earlier
attack *lowered* a second injection's success by 6–20 points (Table 2). No study measuring re-compromise rates of
previously compromised agents was found.

## 11. Reference ZTA implementations actually used (evidence only)

| implementation | decision inputs | rule form | output | source |
|---|---|---|---|---|
| NIST SP 1800-35 (NCCoE, final June 2025, 19 builds) | identity "used most heavily", endpoint compliance, monitoring, threat intel | vendor policy engines per build (Okta/Ivanti, Ping, Entra CA, Zscaler ZPA, Appgate, AWS Verified Access, Google Access Context Manager, PAN, ...) | allow / deny; session continue, revoke or limit | pages.nist.gov/zero-trust-architecture, Vol B, Vol D [S] |
| Google BeyondCorp | user, groups, device certificate, inventory, Trust Inferer tier | first-match ACL (global rules, then service rules); device tier ≥ resource minimum tier | "binary authorization decision" per request; tiers are ordinal (downgrade to remediation-only access) | Ward & Beyer 2014, ;login: 39(6) pp8–9; Osborn et al. 2016, 41(1) pp28–29; Cittadini et al. 2016, 41(4) pp29–30 [S] |
| 800-207A identity-tier policy (Istio AuthorizationPolicy + SPIFFE) | source principal, namespace, method, path, JWT claims | CUSTOM, then DENY, then ALLOW; deny-by-default only if an ALLOW policy exists | allow / deny; mTLS authenticates per **connection**; payload policy "beyond the scope" | 800-207A Fig. 4 p14, pp8–9 ID-SEG-REC-1..5 [S] |
| Envoy + OPA ext_authz, Pomerium PPL, OpenZiti | request attributes | Rego / PPL rules, deny wins; grant-only service policies | allow / deny | vendor docs [S] |
| Bradatsch et al., TrustCom 2023 (arXiv 2402.08299) | 29 trust attributes incl. "Trust History", risk attributes | additive trust score TS vs risk level RL, grant iff TS > RL (Eq. 1–3); subjective-logic variant requires each of user, device, channel to exceed RL | binary | §VI–VII [S] |
| Lukaseder, Halter, Kargl, SICHERHEIT 2020 | subjective-logic opinions from past authentication logs, cumulative fusion | per-role score thresholds in YAML policies | binary | LNI P-301 pp53–66 [S] |

NCCoE builds were evaluated by functional demonstration (use cases A–H, outcomes successful / partial / not
demonstrated), with no latency or false-positive numbers. NCCoE notes that calling the PDP on every request "may be
too great" a cost, so PEPs mostly cache preset rules [S]. Graduation in classic ZTA sits in the inputs (ordinal
tiers, credential lifetime by tier) and in reduced-privilege or step-up responses, not in the PEP gate [S].

## 12. Zero trust for agent traffic in practice (evidence only)

Deployed systems apply a per-hop, per-request, stateless decision: identity, then a default-deny rule over sender,
receiver/tool and arguments (Cedar, CEL, ACLs), then an inline content classifier at a fixed threshold that blocks,
redacts or annotates, plus per-identity rate limits. Only AWS AgentCore carries session history and the caller chain
across hops. No system reasons over topology, carries upstream risk to the next hop, or allocates a risk budget.

| system | identity | authorization | content screening | outputs | quota | history |
|---|---|---|---|---|---|---|
| agentgateway | JWT via MCP OAuth | CEL rules over tool, target, claims | CEL or remote processors | reject, mask, pass | token bucket per key | none |
| Envoy AI Gateway / Agent Router | OAuth, JWT | first-match allow/deny rules | not documented | binary | Envoy rate limits | none |
| AWS AgentCore Gateway + Policy | OAuth JWT or IAM | Cedar, default deny, forbid wins | Bedrock Guardrails scores usable in policy | ALLOW/DENY, suppress output, log-only | count/sum over ≤ 24 h windows | per session |
| Google agent identity + Agent Gateway | SPIFFE, mTLS, DPoP | IAM allow/deny | Model Armor inline | allow / block / redact | not found | none |
| Entra Agent ID + Conditional Access | Entra tokens | if-then rules at token issue | none | grant / block | none | offline agent risk levels |
| LlamaFirewall | n/a | n/a | PromptGuard 2, AlignmentCheck per role | ALLOW / BLOCK / HUMAN_IN_THE_LOOP | n/a | trace replay |

Specs: A2A v1.0 requires authentication on every request (§7.4), leaves authorization "implementation-specific"
(§7.5), and says agents SHOULD rate-limit (§13.4). MCP 2025-11-25 makes authorization optional (OAuth 2.1).
OWASP Agentic Top 10 2026 ASI07/ASI08: mTLS per agent, signed messages, policy-as-code, throttles, circuit breakers.
Roy et al.'s rule-based baseline (arXiv 2610.05782 §6.6) applies "the identical policy set as IF-THEN conditionals";
its rules are not published and it governs device access, not agent messages.

## 13. How the field evaluates (evidence only)

LLM multi-agent defences share one protocol (G-Safeguard 2502.11127, BlindGuard 2508.08127, INFA-Guard 2601.14667):
chain / tree / star / random topologies, scaling to 20–50 agents, CSQA / MMLU / GSM8K tasks, prompt-injection,
tool and memory attacks (InjecAgent, PoisonRAG), metrics ASR at round t and task accuracy. Budgeted placement papers:
MESA (5–7 agents, 8 topologies incl. ER and BA, 145 tasks, top-k% edges, random and oracle bounds, adaptive
attacker, 3–5 seeds) and BRA-Audit (4/8/16 agents, Watts-Strogatz, Barabasi-Albert, Delaunay, audit ratio sweep,
tokens, FPR, 3 repeats). DynaTrust compares against "a Zero-Trust strategy". **ZT-AIR (arXiv 2610.38146)** adds
Verify, Revoke and Micro-segment actions with costs to Cyberwheel and CAGE-4 (50 episodes, paired t-tests).
Cyber-defence convention for graduated actions under cost: CAGE-2 (Monitor, Analyse, Remove, Restore −1, decoys;
penalties per compromised host) and CAGE-4 (action durations Monitor 1 … Restore 5; penalties for failed benign
work). Single-agent anchors: AgentDojo (benign utility, utility under attack, targeted ASR with 95% CIs), ASB.

## 14. Established formulations for BAZT's intended innovations (evidence only)

| innovation | established base | source |
|---|---|---|
| enforcement budget | budget priced on benign traffic; Neyman-Pearson likelihood-ratio threshold set to exhaust the budget | Hua et al., arXiv 2507.15886 §3.1–3.2, Eq. 1–4 |
| audit budget, thresholds | top-x% suspicion audited, threshold as an honest-traffic quantile; exponential budgeting over an episode | Greenblatt et al. 2024; Ctrl-Z arXiv 2504.10374 |
| topology-aware contagion | path-decayed reachability = (truncated) Katz centrality; node value = eigen-drop (NetShield, TKDE 2016); edge blocking (Kimura et al. TKDD 2009; NetMelt CIKM 2012); finite horizon: time-critical influence minimisation (Wang et al. PACMMOD 2025) | as listed |
| online allocation under partial observation | active screening with per-node infection beliefs (Ou et al. AAMAS 2021); restless bandits with Whittle index (Mate et al. NeurIPS 2020) | as listed |
| choosing among graduated actions | maximise α·(expected loss reduction) − β·(control cost) (Poolsappasit, Dewri & Ray, IEEE TDSC 9(1) 2012 §5); damage / response / operational cost classes, respond iff DCost ≥ RCost (Lee et al. 2000/2002) | as listed |
| escalation under uncertainty | multiple optimal stopping, threshold policies on the belief (Hammar & Stadler, IEEE TNSM 19(3) 2022) | as listed |
| safety-usefulness trade-off | AI-Control game, Pareto frontier (Griffin et al., arXiv 2409.07985) | as listed |

Reading: at N = 12 the heterogeneity effect is weak (small-world matches Pastor-Satorras & Vespignani's "no
advantage"); on scale-free graphs it becomes clear from N ≈ 50. The same pilot showed that the existing
`centrality_only` vs `random_capped` comparison confounds *when* an arm spends with *whom* it picks, and that a
fixed `T = 12` truncates spread on larger small-world graphs.
