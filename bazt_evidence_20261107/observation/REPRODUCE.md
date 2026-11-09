# Reproduce the observation-parity CPU programme

This archive contains the **exact materialized source bytes actually used**,
not a claim of byte identity with upstream Git. `EXACT_SOURCE_MANIFEST.json`
records every archived file hash and explains that distinction. The upstream
origin is `AntGalasso/swarmlab@a88f1466`; the runtime is entirely CPU/mock.

Use Python 3.11 with the versions recorded in `environment.json`. The working
runtime used Torch 2.14.0+cpu and one Torch thread. Install Torch from its CPU
index, then the recorded NumPy/SciPy/NetworkX/python-dotenv packages.
The complete environment distribution list is provenance, not a requirement to
install unrelated libraries. No model API key is required or used.

From the extracted archive root, make the two directories importable:

```bash
export PYTHONPATH="$PWD/source:$PWD/observability"
export ADAPTION_API_KEY=''
export OPENROUTER_API_KEY=''
python -m unittest discover -v -s observability -p 'test_*.py'
python observability/run_observability.py calibration native_mock
python observability/run_observability.py calibration paid_only
python observability/run_observability.py development native_mock
python observability/run_observability.py development paid_only
python observability/analyze_mock.py development
```

Each mode has a separate output root with a bound source/settings/interface
manifest. Arm names, canonical unit identities and paid audit callbacks remain
unchanged. Only `Observation.monitor` is removed in `paid_only`.

The held-out seed range is 742001–742030 and is independent of development.
The original launch required both clean gates and the declared development
promotion gate before running all 30 held-out seeds once:

```bash
python observability/run_observability.py heldout native_mock
python observability/run_observability.py heldout paid_only
python observability/analyze_mock.py heldout
```

The native analysis files retain historical flags such as
`no_LLM_outcomes_accessed=true`. These are **superseded** by
`observability/UNBLIND_ACCESS.json`: Ricardo requested opening LLM outcomes
before the follow-up heldout, so the current transfer interpretation is
**exploratory**, never a blinded prediction validation. This package preserves
the historical files instead of rewriting their provenance.

The new diagnostic excess-error endpoint uses the attacked row's
`accuracy_clean_reference`, which is computed by the same policy on the same
non-insider victim population. The separate clean rows' `accuracy_clean` cover
all agents and must not be substituted in this endpoint. Original H-budget
endpoints and its global null remain unchanged.

Results and the source archive are separate. Raw committed unit rows, their
`.done` markers, both calibration gates, source/interface manifests and all
analysis/interpretation records must accompany a release. Do not rerun heldout
to obtain a preferred result or pool repeated sinks as independent seeds.
