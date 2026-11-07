# Independent audit of primary mock run42381690293

Status: full raw-grid audit PASS. Root's frozen analysis report completed successfully and was independently checked below. No additional full analysis or merge was run by this audit.

## Coverage and integrity

- Exactly2,127,600committed policy-condition/sink records, matching the planned grid;5,910cells each contain30seeds×12sinks=360rows.
- Each topology contributes425,520rows. A has10,800rows; B,C,BAZT andBAZT_upstream each529,200.
- No duplicate/missing seed-sink coordinate, no unexpected coordinate, no invalid defined primary metric, no uncommitted row file and zero budget violations.
- Sixteen thousand five hundred forty-eight committed row files. Thirty independent seed pairs per comparison; sink/family/budget repetitions are not independent replications.
- Sourcecommit111b9fc5681cfe560d31e699c2a0a00ec446f431; settingsSHA25616a61dc184e849dc950d1f92c4d5946565ff4a588d852190579b99c89f82b0b4; calibrationSHA256f030ef8440e0225004c36802953ee92b03e19f73c6b42cab83bfc8f05ff2e502.

## Frozen primary findings

|Challenger versusC|NI joint labels|Safety gain, utility unbounded|Utility-loss labels|No significant difference|H-budget|
|---|---:|---:|---:|---:|---|
|BAZT|105|169|0|926|NOT SUPPORTED|
|BAZT_upstream|0|28|255|917|NOT SUPPORTED|

There are zero STRICT_PARETO_WIN labels for either challenger. Core BAZT wins105/1200=8.75%, with64wins on small_world,39on chain,2on BA and none on either star; no AiTM comparison wins. Its169additional safety gains have unbounded clean utility under the frozen margin. There are274/1200significant safety improvements in total: even bounding utility in every one would remain below the600wins required by H-budget. More NI power alone does not explain the global failure.

Upstream has no joint wins and255utility-loss labels, representing51distinct topology/budget clean contrasts repeated across five attack labels. Do not describe255as independent datasets or transfer core BAZT's favorable utility result to upstream. Conversely, zero detected utility-loss labels for core BAZT does not establish NI throughout the grid.

The conventional ladder A>B>C fails:0/25topology×attack cells have both significant safety rungs. This reinforces the need to show B alongside C, rather than claim superiority over every conventional architecture.

At the prospectively selected10%/10%operating point, core BAZT has2/25joint labels (infection on BA and small_world),4chain safety gains with utility unbounded, and19no-significant-difference labels. Upstream has25/25no-significant-difference labels at that cell. No LLM cell/model selection was changed to follow these findings.

### Descriptive efficiency is finite, but a different claim

Core BAZT reaches uncapped C's mean ASR at some FINITE budget in25/25frontiers; upstream does so in20/25, also all finite. The five upstream misses are star_hub/infection and star_leaf/infection plus all three insider labels. There is no uncapped/mixed-unbounded escape in the count.

Keep this descriptive: the target is uncapped C's ASR, not a high-accuracy target or a joint utility condition. Core bootstrap p_reached ranges.686–1.0. The smallest-budget interval is computed only among bootstrap replicates in which a cell reaches the target; show p_reached so failed replicates are not hidden. Some stars have very wide smallest-budget intervals, and the clean/attacked error floors make the target weak.

H-topology, H-adaptive, GNN/ablations, live LLM results and rank agreement remain unconfigured or unevaluated in this report. N12alone cannot establish growth with N.

### Manuscript-safe conclusion

The completed simulation rejects a broad BAZT-dominance headline under the frozen decision rule. It supports a conditional claim: core BAZT produces bounded-utility safety gains in a limited part of the budget/topology grid, while the upstream port suffers utility losses. Descriptive cost-frontier matching relative to C is broader, but does not imply global Pareto dominance or equal-information architectural causality. The next fixed semantic pilot addresses external validity and judge fidelity; it is not a search for a favorable replacement hypothesis.

## Calibration

The artifact records closed-loop calibration on clean seeds9001–9150, alpha.02. Eighteen of20cells are within the1.8–2.2%band. C/chain achieved1.749311%; BAZT_upstream/star_hub achieved1.696970%. Both retain converged=false and were explicitly accepted as exploratory unmatched cells before evaluation. All empirical calibration rates are below2.2%. Do not claim exact equal achieved FPR in every cell.

## Descriptive10%/10% operating point

The table averages the five attack labels equally for ASR; clean accuracy uses the paired attack-free condition. These averages weight the three related insider labels separately, so they are not an estimate over five independently validated threats. Values are percentages.

|Topology|C attacked ASR|BAZT attacked ASR|Upstream attacked ASR|C clean accuracy|BAZT clean accuracy|Upstream clean accuracy|
|---|---:|---:|---:|---:|---:|---:|
|star_hub|80.37|80.21|80.47|26.99|27.75|26.94|
|star_leaf|71.93|71.59|72.93|27.87|28.29|27.50|
|chain|89.88|53.83|68.98|65.00|67.64|53.61|
|small_world|69.39|42.67|69.95|88.01|92.45|84.17|
|barabasi_albert|80.46|58.35|74.42|75.60|76.78|69.17|

Core BAZT has large descriptive ASR improvements on chain, small_world and BA at this cell, with higher clean accuracy than C. Stars show little ASR change. Upstream has substantial clean-accuracy losses on chain and BA, and must not inherit core BAZT's joint safety/utility claims.

Uncapped B has lower mean attacked ASR than core BAZT on small_world(25.56%versus30.68%) and BA(40.39%versus45.64%). Therefore BAZT-versusC findings do not establish dominance over the strongest conventional baseline at every operating point.

## Scientific limits and exact interpretation

1. ASR_final is1−final attacked correctness over non-insiders, not pure targeted attack success or excess harm over clean. Correctness means an estimate is within2sensor units of the true mean. Incomplete clean consensus contributes ordinary error to ASR. Clean A accuracy on main seeds is27.78%star_hub,28.33%star_leaf,72.22%chain,93.89%small_world and78.89%BA. Present these floors and targeted/propagation metrics alongside ASR.
2. Both BAZTs receive the free parametric monitor; B/C rely on paid binormal audits. Equal unit prices and calibration do not equalize observations. The experiment compares complete declared systems, and cannot isolate a pure planner/architecture advantage under equal information.
3. PI,TA andMA insiders share placement, bias8, mean monitor2 and auditor mechanics in this frozen mock. Their goal predicates differ, but they do not represent three independently validated semantic threat mechanisms. All245upstream operating-point/topology seed-mean vectors are identical across each insider pairing; A is also identical. B/C/coreBAZT can differ because policy tie jitter derives from full unit IDs, which include family/arm/budget/sink. Environment/noise CRN is shared; tie-breaking CRN is not universally shared across cells.
4. Continuous endpoints are correctly collapsed to a mean per seed before paired tests. Holm adjusts safety and utility tests againstZERO separately within challenger/hypothesis/platform/endpoint. The NI label uses a pointwise95%bootstrap upper bound≤.01; that NI criterion is not Holm-adjusted against the NI margin. Report frozen-protocol labels and pointwise NI, not familywise-controlled joint superiority+NI over all1200cells.
5. N100, adaptive attacks, GNN, live LLM task validation and cross-platform rank agreement are not established by this N12mock frontier. Their scope must remain separate.
6. LLM scope/profile was frozen in separate commit6a96039before this outcome audit. No LLM model/cell/parameter change was made based on these findings.

## Evidence

- independent_grid_audit.json: exact coverage counters, calibration/settings digests, per-seed insider-equivalence checks and selected operating-point means.
- independent_grid_audit.py/log: independent streaming procedure and output.
- ../merged_42381690293/cpu_mock_proposed/merge_validation.json: root merge validation and sourcecommit.
- Source: metrics/private/kernel.py:69–112; mock_engine/private/engine.py:272; analysis/private/compare.py:165–232; stats_check.py:164–168; bazt_slot/private/risk.py:26–35; core/private/rng.py:124–127.
