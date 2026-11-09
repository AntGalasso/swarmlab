# Fixed two-graph observability extension — completed

The extension is complete and independently audited: 25,920 raw evaluation
records, 30 independent seeds per graph, twelve repeated sinks, two observation
interfaces and the original nine arm/budget cells. There are zero missing IDs,
duplicates, orphans or budget violations. All 7,200 A/B/C negative-control rows
are identical between modes except declared provenance events.

All sixteen clean calibration cells are within the 1.8–2.2% band, and B/C
thresholds match between modes for each graph. Heldout false-positive rates are
not asserted to be equal. The estimand includes each interface's recalibration
protocol; it is not a fixed-threshold ablation.

| Graph | Change in core-vs-C same-victim excess error | Pointwise 95% CI | Holm-2 p |
|---|---:|---:|---:|
| Small-world | +43.813 pp | [33.131, 54.092] pp | 0.000020 |
| Star-hub | approximately 0 pp | [-0.985, 1.010] pp | 1 |

In small-world, the mean BAZT-minus-C contrast changes from -34.621 pp with the
free monitor to +9.192 pp with paid-only information. The paid-only contrast's
pointwise interval includes zero; this is not proof that C beats core after
correction. In star-hub the contrast is -1.263 pp under both interfaces.

Both fixed graphs are retained. These are secondary, exploratory extension
results fixed after BA and LLM outcomes were observed; they do not confirm the
original H-topology hypothesis or rescue the original H-budget global null.
Upstream and uncapped comparisons remain descriptive. No provider calls were
made.

The archive includes all extension code, configuration, native raw calibration
and evaluation records, checkpoints, manifests, gates, analysis and execution
logs. Native bazt_lab/bazt dependencies are the unchanged exact source from the
original BA source archive, whose hash is recorded in PACKAGE_RECEIPT.json.
The original BA source/results archives and scientific files remain untouched.
