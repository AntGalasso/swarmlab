"""Loads the pinned revisions from the Hugging Face Hub. Runs only with BAZT_NETWORK_TESTS=1.

    BAZT_NETWORK_TESTS=1 python -m pytest bazt_lab/features/llm_data/tests/test_network_smoke.py -q

The second half re-runs the same draw in a fresh interpreter with HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1, which
must be served from the local cache filled by the first half.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bazt_lab.features.llm_data import public as llm_data
from bazt_lab.features.llm_data.tests.conftest import REV

pytestmark = pytest.mark.skipif(os.environ.get("BAZT_NETWORK_TESTS") != "1",
                                reason="network smoke test; set BAZT_NETWORK_TESTS=1")

REPO = Path(__file__).resolve().parents[4]
SETTINGS = {"llm_data": {"revisions": REV}}
EXPECTED = {"mmlu": ("cais/mmlu", "all", "test", 14042), "csqa": ("tau/commonsense_qa", "default", "validation", 1221),
            "gsm8k": ("openai/gsm8k", "main", "test", 1319)}


@pytest.mark.parametrize("task", sorted(EXPECTED))
def test_pinned_revision_loads_and_items_are_well_formed(task):
    dataset, config, split, _ = EXPECTED[task]
    got = llm_data.sample(task, 1, 4, SETTINGS)
    assert len(got) == 4 and got == llm_data.sample(task, 1, 4, SETTINGS)
    for it in got:
        assert (it.source.dataset, it.source.config, it.source.split, it.source.revision) == (
            dataset, config, split, REV[task])
        assert it.question and it.answer and it.id.startswith(task + ":")
        assert llm_data.check_answer(it, it.answer) is True
        if task != "gsm8k":
            assert len(it.choices) == (4 if task == "mmlu" else 5) and llm_data.render_question(it).count("\n") >= 4
    assert got != llm_data.sample(task, 2, 4, SETTINGS)


def test_offline_rerun_is_served_from_the_cache():
    code = ("import json; from bazt_lab.features.llm_data import public as d; "
            "from bazt_lab.features.llm_data.tests.conftest import REV; "
            "print(json.dumps({t: [i.id for i in d.sample(t, 1, 4, {'llm_data': {'revisions': REV}})] "
            "for t in ('mmlu', 'csqa', 'gsm8k')}))")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
                         check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    online = {t: [i.id for i in llm_data.sample(t, 1, 4, SETTINGS)] for t in EXPECTED}
    assert json.loads(out.stdout.splitlines()[-1]) == online


def test_split_sizes_are_the_expected_ones():
    from bazt_lab.features.llm_data.adapters import hf

    for task, (dataset, config, split, size) in EXPECTED.items():
        rows = hf.draw(dataset, config, split, REV[task], seed=0, k=size)
        assert len(rows) == size
        with pytest.raises(ValueError):
            hf.draw(dataset, config, split, REV[task], seed=0, k=size + 1)
