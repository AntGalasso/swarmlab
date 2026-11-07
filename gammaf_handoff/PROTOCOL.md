# Frozen voting protocol and reproduction additions

The voting/selection analysis was developed in Ricardo's private research branch.
This handoff preserves its source, source pins and reference JSONs. It adds the
single-command runner, bundle checksums, standard-library projection checks,
per-question exports, and the CSQA alphabet-aware sensitivity check.

## Original data access and development split

Use GAMMAF commit `53be3249ebfa631d8e960c9fb1c931d526654697`, Information Sciences
experiment 1. MMLU is development; MMLU-Pro, CSQA and GSM8K evaluate the fixed
selections. Before this calculation, the released multitask summary, MMLU records
and disabled controls for all tasks had already been accessed. This is a new
calculation on public data, not a blind replication.

The original voting protocol was committed at
`62d6cfaae4cb5709ce5ad97829a898c7a06fcd9e`. MMLU selections and source identities
were frozen at `42a90a2bc8c87e9e267e2a91d239ebfcd297dfa3` before accessing the
named detectors' raw validation records. The independent strict-format check was
specified later, at `4d5892064b8fff2f88271e06b0b0b73b21a512d2`, after the first
discordance exposed braced answer strings. The final transferred analysis snapshot
is `a8c0cb5f2c52ce858fd7cf47a6b3a3dfbbea2b72`. These are provenance references in
the private source repository, not commits teammates need to access to run this package.

## Voting and selection

Use each debate's last executed round. Reconstruct (a) plurality over all ten
terminal voters and (b) plurality over only unflagged terminal voters. Preserve
original answer strings; use the upstream first-occurrence rule for tied plurality
counts. Compare (a) with the saved correctness label and (b) with the saved final
answer. For letter tasks correctness is case-insensitive; GSM8K uses exact strings.
Check every published task/method/topology correctness count against the released
aggregate file. Check all question IDs/gold labels and the repeated four-graph grid.

Select the highest equal-graph MMLU accuracy separately under (a) and (b). Resolve
exact method ties by Python string order and retain the complete tie set. Do not
substitute a more favourable detector pair after viewing validation. On validation,
calculate delivered accuracy for the two fixed choices. Average four graphs within
question, then average questions within task, and weight the three tasks equally.
Resample question IDs 20,000 times within task, retaining all detectors and four
graphs together. The fixed NumPy RNG seed is `202611071017`. This estimates sampling
sensitivity conditional on the fixed tasks, topologies and saved model outputs.

## Excluded-vote stress test

On a copy of each terminal record, keep flags, unflagged votes, order and the gold
answer fixed. Replace all excluded votes in turn with each distinct observed
answer, the gold answer, and one fresh candidate. For numeric output the fresh
candidate is a distinct integer; MMLU-Pro uses A-J; the original other-letter rule
uses A-D. Retain that frozen candidate set, including for CSQA; do not interpret
the resulting sensitivity as an exhaustive search over all possible strings.
Check whether any replacement changes all-voter correctness and verify that the
eligible-voter answer remains identical throughout. No live trajectory is simulated.

## Format and no-tie sensitivity checks

The original SQL script uses exact A-D answers for MMLU and CSQA, A-J for MMLU-Pro,
and canonical integer strings for GSM8K. It requires unique plurality winners in
both populations. This narrower original subset reproduces 36/4,177 disagreements.

During handoff preparation, source data and the public CSQA loader confirmed five
options, A-E. An additional check changes only the CSQA alphabet to A-E and yields
41/4,301. It is labelled an additional reproduction check, not an original frozen
test. The unmodified SQL script and reference result remain included. All primary
scores, selections, validation intervals and stress-test definitions remain intact.

## Detection-credit analysis

`detection_credit/README.md` documents the separate, earlier exploratory analysis.
Its projection contains 14,400 round rows; do not add that count to the 11,200
terminal decisions as independent samples. It compares released third-round ADR
with known-adversary flag coverage at the last executed round, and reports all 15
MMLU detector pairs with joint question resamples and simultaneous intervals.
