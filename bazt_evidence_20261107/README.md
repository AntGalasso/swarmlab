# BAZT completed evidence review — 7 November 2026

This package prepares the team's conventional-control benchmark evidence for review. It changes no runtime implementation, existing result, calibration threshold or active workflow, and contains **no private manuscript**. It performs no model requests. N100 remains António's separate PR10 task; H4/causal remain paused.

## Findings and claim boundaries

* Original main numerical benchmark: 2,127,600 committed records, 5,910 cells, 30 independent seeds with 12 repeated sinks. Original global H-budget **unsupported**: native 105/1200 joint labels, upstream 0/1200. Related insider labels, free monitor asymmetry and weak star clean floors remain limitations. Native conditional gains are not transferred to upstream.
* Recalibrated observation-contract diagnostic: BA native-minus-C excess error changes by +21.52 pp [12.25,31.06]. Removing only the free monitor plus recalibrating changes the native advantage from −27.27 to −5.76 pp. It is not a fixed-threshold or matched-heldout-FPR intervention. A/B/C controls match exactly.
* Secondary exploratory extension: small-world +43.81 pp [33.13,54.09], Holm-2 p≈.000020; star-hub approximately zero [−.98,1.01], p=1. Both fixed graphs are retained; no original H-topology confirmation.
* Completed semantic pilot: 160 committed units/320 rows, ten independent seed clusters, 2 conversations per arm/condition/seed. All clean accuracy90%; all uncapped incorrectness11.36%; capped native10.91% vs C/upstream11.36%. Difference is one victim answer in one seed, original Holm p=1. Identity A has0/20 final targeted successes. The challenge scarcely separates the stacks: this is not equivalence or general containment.
* ALL cross-platform comparisons are **exploratory**. Coordinating analysts had opened LLM outcomes before this follow-up, despite historical worker-local blind flags. The later worker lock is not study-level blindness. Removing the free monitor did not improve even exploratory rank prediction (.25 native versus .75 paid-only normalized distance). Historical declarations are preserved and explicitly superseded in `semantic/COORDINATION_DISCLOSURE.json`.

Excess error means attacked final incorrectness minus matched clean incorrectness on the same non-insider victim population, unclamped. It is not independent clean utility or targeted success. Sinks, budget cells and conversations do not increase seed counts. No new outcome is substituted for the original frozen analysis.

## Inspect without inference

Run `python bazt_evidence_20261107/verify_evidence.py` from the repository root. It checks all packaged byte hashes, raw archive members, semantic grid/IDs, retained unknown reservation and corrected access disclosure. No API key or third-party dependency is needed.

`primary/` includes the immutable profile/approval manifest, strict shard merger and independent audit/extractors (the unchanged historical audit is retained as .py.txt; Python can run that path directly), full original compressed analysis, descriptive table and claim audits. `observation/` contains the exact materialized CPU source archive, complete calibration/development/heldout raw row/done/state archives and the independently audited two-graph extension. `semantic/` retains all320 canonical committed rows, the untouched original frozen analysis, completeness/ledger review and explicit post-lock metric reconstruction disclosure. Full provider reply/cache/ledger lineage remains in the authoritative private Actions artifact, not duplicated as a second API campaign.

## Reproduce the controlled numerical follow-up

Extract `observation/observation_parity_exact_source_20261107.tar.gz` in an isolated directory; follow its `REPRODUCE.md`, using Python3.11 and the recorded CPU dependencies. The exact source bytes include the documented extra terminal LF in208runtime files; no other runtime code difference was found. Do not silently substitute current-main source. The archive includes its source manifest, environment and engine tests. `observation/REPRODUCTION_CHECK.json` records a successful fresh-extraction check. Extract the extension archive separately; its package receipt binds it to the same native source archive. Provider keys must remain empty. Reproduction uses separate output roots; do not overwrite historical records.

The original primary can be reproduced in a detached worktree of team source a88f146675134133b85ae7eaf16d38048096c7c6 with the frozen primary profile copied into `bazt_headline/headline_cpu_proposed.toml`. The original driver is `run_experiments.py`: first `--run --block mock_calibration`, then validated thresholds, then `--run --block mock_frontier` in five disjoint topology shards. Consult `primary/merge_shards.py --help` and the preserved profile/approval manifest. The exact production launch and environment are also retained in the original workflow below. Do not overwrite the existing primary or retune the protocol. The committed summaries are already complete; reproduction is optional.

## Authoritative runs and source lineage

- Primary mock: ricardofitas/swarm-adaption-hackathon run42381690293, launch111b9fc5681cfe560d31e699c2a0a00ec446f431. Five final artifact names are `bazt-headline-evaluation-{barabasi_albert,chain,small_world,star_hub,star_leaf}`. All five byte-identical final raw ZIPs are now also durably preserved here as ordered8MiB parts, with original archive/part SHA256 and Git blob hashes in primary/RAW_ARCHIVES.json. Reassemble, for example, with `python bazt_evidence_20261107/reassemble_primary.py chain --output-dir /tmp/bazt-raw`. This is storage chunking only; all original manifests, row/done pairs and contents are preserved. No primary merge or scientific analysis was rerun.
- Reviewed BA: private commit2ff8168ed29208f24bbf0ca1871cced348bed373. Full raw/source archives are included here.
- Two-graph extension: private commit0bc8b1c4837c0a9624c7a395322372b850b13ff8. Full raw archive is included here.
- Semantic run42391846181, source358bcd90a7e0d5e6c134a947409d9fd3c55ed817, artifact13583835617 SHA256ba2d812009de8a3e30973cf762fd0f3eee067a814194b898518d4cdc6d12849c. All320 scientific rows are included here. Retained unknown998 is fully counted; no currency mapping is asserted.

This is evidence/code preparation for team review. No hackathon submission, email, chat message, or private manuscript publication is performed.
