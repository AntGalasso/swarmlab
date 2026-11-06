"""The Hub adapter against the real `datasets` library, on an in-memory split (no network)."""
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bazt_lab.features.llm_data import public as llm_data
from bazt_lab.features.llm_data.adapters import hf
from bazt_lab.features.llm_data.tests.conftest import REV, ROWS

datasets = pytest.importorskip("datasets")

REPO = Path(__file__).resolve().parents[4]
N = 50
GOLDEN_SEED_7_K_8 = [10, 45, 24, 46, 20, 12, 37, 0]      # datasets 5.1.0, numpy 2.4.6, 50 rows


@pytest.fixture
def hub(monkeypatch):
    """hf._datasets() returns a stand-in whose load_dataset records its arguments and yields a real Dataset."""
    seen = []
    split = datasets.Dataset.from_list([{"question": f"q{i}", "n": i} for i in range(N)])

    def load_dataset(path, name=None, **kwargs):
        seen.append((path, name, kwargs))
        return split

    monkeypatch.setattr(hf, "_datasets", lambda: types.SimpleNamespace(load_dataset=load_dataset))
    hf._split.cache_clear()
    yield seen
    hf._split.cache_clear()


def test_draw_is_shuffle_seed_then_select_k_and_tags_the_original_row(hub):
    rows = hf.draw("org/data", "cfg", "test", "r" * 40, seed=7, k=8)
    ref = datasets.Dataset.from_list([{"question": f"q{i}", "n": i} for i in range(N)]).shuffle(seed=7).select(range(8))
    assert [r["n"] for r in rows] == [r["n"] for r in ref]               # the spec's definition, literally
    assert [r["_row"] for r in rows] == [r["n"] for r in rows]            # `_row` is the pre-shuffle position
    assert [r["_row"] for r in rows] == list(np.random.default_rng(7).permutation(N)[:8])
    assert all(r["question"] == f"q{r['n']}" for r in rows)


def test_the_draw_is_pinned_against_library_drift(hub):
    assert [r["_row"] for r in hf.draw("org/data", "cfg", "test", "r" * 40, seed=7, k=8)] == GOLDEN_SEED_7_K_8


def test_load_dataset_is_called_with_the_pinned_revision_and_no_token(hub):
    hf.draw("cais/mmlu", "all", "test", "a" * 40, seed=1, k=2)
    assert hub == [("cais/mmlu", "all", {"split": "test", "revision": "a" * 40})]


def test_a_split_is_resolved_on_the_hub_once_per_process_and_revision(hub):
    """A run draws once per seed and task. Each `load_dataset` call asks the Hub to resolve the revision again, so
    thousands of draws would be thousands of requests (rate limits, a transient error killing a long run)."""
    a = hf.draw("org/data", "cfg", "test", "a" * 40, seed=1, k=5)
    b = hf.draw("org/data", "cfg", "test", "a" * 40, seed=2, k=5)
    hf.draw("org/data", "cfg", "test", "b" * 40, seed=1, k=5)            # another revision is another split
    hf.draw("org/data", "cfg", "validation", "a" * 40, seed=1, k=5)      # so is another split
    assert [(kw["split"], kw["revision"]) for _, _, kw in hub] == [
        ("test", "a" * 40), ("test", "b" * 40), ("validation", "a" * 40)]
    assert [r["_row"] for r in a] != [r["_row"] for r in b]              # the cached split still draws per seed
    assert [r["_row"] for r in hf.draw("org/data", "cfg", "test", "a" * 40, seed=1, k=5)] == [r["_row"] for r in a]


def test_a_failed_load_is_not_remembered(monkeypatch):
    """A transient Hub error must not poison the process: the next draw asks the Hub again."""
    attempts = []
    split = datasets.Dataset.from_list([{"question": f"q{i}"} for i in range(N)])

    def flaky(path, name=None, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("Hub unreachable")
        return split

    monkeypatch.setattr(hf, "_datasets", lambda: types.SimpleNamespace(load_dataset=flaky))
    with pytest.raises(ConnectionError):
        hf.draw("org/flaky", "cfg", "test", "f" * 40, seed=1, k=3)
    assert len(hf.draw("org/flaky", "cfg", "test", "f" * 40, seed=1, k=3)) == 3 and len(attempts) == 2


def test_rows_typed_as_the_pinned_cards_declare_them_become_items(monkeypatch):
    """Assumption about `datasets`: with the features the three cards declare at the pinned revisions, a row reaches
    the item builders as the shapes they need (CSQA `choices` as {"label": [...], "text": [...]}, MMLU `answer` as an
    int from a ClassLabel). A library that returned CSQA choices as a list of dicts would fail here, not mid-run."""
    V, S = datasets.Value, datasets.Sequence
    features = {
        "mmlu": datasets.Features({"question": V("string"), "subject": V("string"), "choices": S(V("string")),
                                   "answer": datasets.ClassLabel(names=list("ABCD"))}),
        "csqa": datasets.Features({"id": V("string"), "question": V("string"), "question_concept": V("string"),
                                   "choices": S({"label": V("string"), "text": V("string")}),
                                   "answerKey": V("string")}),
        "gsm8k": datasets.Features({"question": V("string"), "answer": V("string")}),
    }
    splits = {task: datasets.Dataset.from_list([ROWS[task](i) for i in range(6)], features=features[task])
              for task in features}
    monkeypatch.setattr(hf, "_datasets",
                        lambda: types.SimpleNamespace(load_dataset=lambda path, name=None, **kw: splits[
                            {"cais/mmlu": "mmlu", "tau/commonsense_qa": "csqa", "openai/gsm8k": "gsm8k"}[path]]))
    hf._split.cache_clear()
    settings = {"llm_data": {"revisions": REV}}
    try:
        for task, letters_of in (("mmlu", "ABCD"), ("csqa", "ABCDE"), ("gsm8k", None)):
            got = llm_data.sample(task, 3, 6, settings)
            assert sorted(it.id for it in got) == sorted(f"{task}:{i}" for i in range(6))
            for it in got:
                i = int(it.id.split(":")[1])
                want = ROWS[task](i)
                assert it.question == want["question"]
                if task == "mmlu":
                    assert (it.choices, it.answer) == (tuple(want["choices"]), letters_of[want["answer"]])
                if task == "csqa":
                    assert (it.choices, it.answer) == (tuple(want["choices"]["text"]), want["answerKey"])
                if task == "gsm8k":
                    assert it.answer == str(i * 1000)
    finally:
        hf._split.cache_clear()


def test_k_beyond_the_split_is_an_error(hub):
    with pytest.raises(ValueError, match="k"):
        hf.draw("org/data", "cfg", "test", "r" * 40, seed=1, k=N + 1)


def test_env_file_is_loaded_without_overriding_the_process_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("BAZT_T_FROM_FILE=file\nBAZT_T_BOTH=file\n", encoding="utf-8")
    monkeypatch.setattr(hf, "ENV_PATH", env)
    monkeypatch.setenv("BAZT_T_BOTH", "process")
    monkeypatch.delenv("BAZT_T_FROM_FILE", raising=False)
    try:
        hf._load_env()
        assert os.environ["BAZT_T_FROM_FILE"] == "file" and os.environ["BAZT_T_BOTH"] == "process"
    finally:
        os.environ.pop("BAZT_T_FROM_FILE", None)


def test_a_missing_env_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "ENV_PATH", tmp_path / "absent.env")
    hf._load_env()


def test_importing_the_feature_does_not_import_datasets():
    """Dry runs and the architecture gate import public.py; `datasets` loads on the first Hub draw only."""
    code = "import sys; import bazt_lab.features.llm_data.public; print('datasets' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300,
                         check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-1] == "False"
