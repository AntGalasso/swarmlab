# GAMMAF evaluation audit: team reproduction

This is Team Vittoria's offline reanalysis of released GAMMAF experiment 1 records.
GAMMAF is the independent work of Pablo Mateo-Torrejón and Alfonso Sánchez-Macián.
All public inputs are pinned to [GAMMAF commit 53be324](https://github.com/pmateo-uc3m/GAMMAF/tree/53be3249ebfa631d8e960c9fb1c931d526654697).
The task is to independently reproduce the voting and detection-credit findings,
inspect an example, and return the generated archive. Both modes make zero model calls.

## Start here: no installation, GPU or API key

Use Python 3.11 or 3.12. From a **new checkout**, run:

```bash
git clone --branch handoff/gammaf-reproduction-20261107 --single-branch https://github.com/AntGalasso/swarmlab.git swarmlab-gammaf
cd swarmlab-gammaf
python3 gammaf_handoff/run.py
```

Authenticate to the private team repository using your normal GitHub access.
The audit itself needs no credentials. If the branch has been merged, the same
`python3 gammaf_handoff/run.py` command works on the updated main branch.
Do not install the full swarmlab/G-Safeguard environment for this task.

The command verifies the package checksums, recomputes all 11,200 terminal voting
decisions from the bundled compact projection, checks detector/task/topology totals,
repeats the excluded-vote perturbation test, checks both strict-format subsets,
and reproduces the detection-credit analysis. It checks the frozen MMLU selection
and the validation point estimates. It uses Python's standard library only.

Outputs go to a new `gammaf_handoff/_runs/<UTC timestamp>/` directory. A successful
run prints `PASS` and the exact path of **reproduction.zip**. This quick mode checks
the bundled projection; it does not independently authenticate public source bytes
or recompute the NumPy bootstrap intervals. Use the next mode for those checks.

## Full reproduction from the public originals

This is the preferred independent confirmation. It downloads about 122 MB of
public JSON into an ignored cache, verifies every input's pinned Git blob and byte
count, rebuilds both voting populations, recomputes the bootstrap intervals, and
checks the independent SQL reconstruction and detection-credit projection.
It requires public internet access, NumPy and CPU only.

```bash
python3 -m venv .venv-gammaf
.venv-gammaf/bin/python -m pip install -r gammaf_handoff/requirements.txt
.venv-gammaf/bin/python gammaf_handoff/run.py --from-source
```

On Windows, use `.venv-gammaf\Scripts\python.exe` instead of
`.venv-gammaf/bin/python`. No activation step is necessary.

An existing checkout/cache of the pinned public revision can be used directly:

```bash
.venv-gammaf/bin/python gammaf_handoff/run.py --from-source --raw /path/to/GAMMAF
```

Missing files are downloaded; present files with incorrect hashes fail rather than
being overwritten. `--workers 1` reduces download concurrency when needed.
The raw path is the **repository root**, not the `results` subdirectory.
Use `--out <new-directory>` to choose a different output location. Existing output
directories are refused so independent runs do not overwrite one another.

## What to check and return

1. Run the quick command, and preferably the full public-source command.
2. Inspect `discordant_cases.json`; CSQA / BlindGuard / item 1845 / chain is a
   simple case with valid answers and no plurality tie. The delivered answer is C
   and is correct; the all-voter score reports it wrong.
3. Inspect `VOTING_CHECK.json` and `CHECKS.json`. Confirm the reference values below
   and whether the receipt says `bundled_projection` or `public_source_rebuild`.
4. Return the generated **reproduction.zip**, plus your OS/Python version and any
   failure. Do not return the downloaded public cache or any API keys.

The archive contains the verification receipts, all observed discordant cases,
and `paired_questions.csv`, with one row per reserved-task question/topology for
the two fixed selections. Full mode also includes the regenerated voting export,
selection, bootstrap results and SQL receipt. Four topologies repeat each question:
the CSV rows are not independent observations for statistical inference.

## Expected results

| Check | Expected |
|---|---:|
| Terminal method-debates, including disabled controls | 11,200 |
| Debates with one of the six detectors | 9,600 |
| Observed correctness-label disagreements | 227 / 9,600 |
| Score correct / delivered wrong | 156 |
| Score wrong / delivered correct | 71 |
| Disabled controls: disagreements / excluded-vote sensitivity | 0 / 0 across 1,600 debates |
| Frozen excluded-vote stress test can change score, keeping delivery fixed | 1,786 / 9,600 |
| Original narrow-alphabet, no-tie subset | 36 / 4,177 disagreements |
| Dataset-alphabet, no-tie subset, including CSQA option E | 41 / 4,301 disagreements |
| MMLU released-score choice | XG-Guard: 328 / 400 |
| MMLU delivered-score maximisers | TAM and XG-Guard: 328 / 400 each |
| Alphabetical tie rule's delivered-score choice | TAM |
| Reserved-task delivered accuracy: TAM minus XG-Guard | -1.75 percentage points |
| Frozen paired-bootstrap 95% interval, full mode | [-3.25, -0.33] percentage points |
| MMLU disabled detector: released round-3 ADR / actual flags ADR | 8% / 0% |

**The selection result is a tie-sensitive negative result.** XG-Guard also maximises
development delivered accuracy. Keeping that common maximiser gives zero change.
These data do not show that correcting the scoring rule selects a better defence.

The original SQL sensitivity script used A-D for both MMLU and CSQA, excluding E
from CSQA. Its frozen 36/4,177 receipt is retained verbatim. `compact_check.py`
also checks A-D for MMLU, A-E for CSQA, A-J for MMLU-Pro, and integer strings for
GSM8K; that additional check yields 41/4,301 (26 false credits, 15 missed credits).
Neither robustness subset includes tied plurality winners. The original excluded-vote
stress test and all primary results remain frozen; its candidate set is documented
in [PROTOCOL.md](PROTOCOL.md).

## Interpretation and files

Read [RESULTS.md](RESULTS.md) for the evidence and claim boundaries,
[PROTOCOL.md](PROTOCOL.md) for selection/resampling rules and prior data access,
and [PROVENANCE.json](PROVENANCE.json) for original analysis and public-source pins.
`gamma_votes.py` and `verify_votes.py` preserve the original analysis source.
`expected/` contains its frozen reference results; `detection_credit/` preserves
the earlier detection-credit audit as a separate analysis.

This is a replay of external published model outputs. It is separate from the
team's BAZT simulator/live-model experiments. The package provides no evidence of
BAZT superiority or deployment safety. It does not claim human team reproduction
before a teammate actually runs it. The underlying GAMMAF material is attributed
above and covered by the bundled [upstream Apache 2.0 license](UPSTREAM_LICENSE).
