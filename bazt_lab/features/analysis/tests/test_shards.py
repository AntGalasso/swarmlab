"""analysis on the layouts core.CheckpointStore leaves on disk: worker shards (units/<shard>/, compute.workers > 1), a
block resumed with another worker count, and rows files that were never committed (a batch a crash cut before its done
list, in a shard no later run writes to; a unit still running or abandoned in another shard).

The rows are written through the real store, so these tests follow its layout and done markers."""
import json

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.analysis import public as analysis
from bazt_lab.features.analysis.private import cli
from bazt_lab.features.analysis.private import table as tb
from bazt_lab.features.analysis.tests import synth

U = "uncapped"
CAPPED = [(0.1, 0.1), (0.2, 0.2)]
BATCH = 25                     # units per append_batch (mock side)
PER_UNIT = 4                   # rows per unit (LLM side: one unit is a seed run, a row is one conversation)


def mock_rows():
    asr = {"A": 0.6, "B": 0.5, "C": 0.4, "BAZT": 0.25}
    arms = {a: (lambda cell, topo, n, a=a: asr[a]) for a in asr}
    return synth.episodes(arms, cells_by_arm={"A": [(U, U)], "B": [(U, U)], "C": [(U, U)] + CAPPED, "BAZT": CAPPED},
                          topologies=("chain", "barabasi_albert"), seeds=range(1, 13), sinks=range(2), noise_sd=0.01)


def llm_rows():
    return synth.episodes({"C": lambda *a: 0.4, "BAZT": lambda *a: 0.3}, block="llm_capped", platform="llm",
                          cells=[(0.1, 0.1)], topologies=("random_directed",), seeds=range(1, 6),
                          conversations=range(4), tasks=("mmlu",), backbones=("gpt",), noise_sd=0.01)


MOCK, LLM = mock_rows(), llm_rows()


class Stores:
    """One CheckpointStore per (block, shard), opened once, as each worker process holds its own."""

    def __init__(self, root):
        self.root, self.open = root, {}

    def __call__(self, block, shard):
        if (block, shard) not in self.open:
            self.open[block, shard] = core.CheckpointStore(self.root / block, shard=shard)
        return self.open[block, shard]


def commit_batches(stores, rows, shard_of):
    """The mock runner: one append_batch per BATCH units (one row each); batch k is committed in shard shard_of(k)
    (None: units/ itself, the single-writer layout)."""
    for block, items in synth.by_block(rows).items():
        for k, start in enumerate(range(0, len(items), BATCH)):
            stores(block, shard_of(k)).append_batch(
                {f"u{i:05d}": [synth.record(items[i])] for i in range(start, min(start + BATCH, len(items)))})


def commit_units(stores, rows, shard_of):
    """The LLM harness: the rows of a unit appended, then the unit marked done; unit k in shard shard_of(k)."""
    for block, items in synth.by_block(rows).items():
        for k, start in enumerate(range(0, len(items), PER_UNIT)):
            store, uid = stores(block, shard_of(k)), f"u{start:05d}"
            store.append_rows(uid, [synth.record(r) for r in items[start:start + PER_UNIT]])
            store.mark_done(uid)


def single_writer(root):
    stores = Stores(root)
    commit_batches(stores, MOCK, lambda k: None)
    commit_units(stores, LLM, lambda k: None)


def two_workers(root):
    stores = Stores(root)
    commit_batches(stores, MOCK, lambda k: f"w{k % 2}")
    commit_units(stores, LLM, lambda k: f"w{k % 2}")


def resumed_with_workers(root):
    """A block started with one writer (units/), stopped half way, resumed with --workers 2 (units/w0, units/w1)."""
    stores = Stores(root)
    n_batches, n_units = -(-len(MOCK) // BATCH), -(-len(LLM) // PER_UNIT)
    commit_batches(stores, MOCK, lambda k: None if k < n_batches // 2 else f"w{k % 2}")
    commit_units(stores, LLM, lambda k: None if k < n_units // 2 else f"w{k % 2}")


def stale_shard_leftovers(root):
    """A first launch with --workers 3: worker w2 crashed between the rows file of its batch and the done list (an
    orphan), and left a unit half written. The next launch, with --workers 2, ran those units again in w0 and w1.
    Nothing writes to w2 again, so its own orphan clean-up never runs there."""
    stores = Stores(root)
    stores("mock_frontier", "w2").append_batch({f"u{i:05d}": [synth.record(MOCK[i])] for i in range(BATCH)})
    next((root / "mock_frontier" / "units" / "w2").glob("batch-*.done")).unlink()       # the commit never happened
    stores("llm_capped", "w2").append_rows("u00000", [synth.record(r) for r in LLM[:2]])  # never marked done
    two_workers(root)


def report(root):
    """(the report as JSON text without results_dir and uncommitted_files, uncommitted_files per platform, the report)"""
    r = analysis.analyze(root, synth.settings(), out=root / "report.json")
    plain = json.loads(json.dumps(r))
    plain["meta"].pop("results_dir")
    uncommitted = {p: plain["meta"]["platforms"][p].pop("uncommitted_files", None) for p in ("mock", "llm")}
    return json.dumps(plain, sort_keys=True), uncommitted, r


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("single_writer")
    single_writer(root)
    return report(root)


def test_the_single_writer_reference_is_not_vacuous(reference):
    _, uncommitted, r = reference
    assert r["meta"]["platforms"]["mock"]["rows"] == len(MOCK) and r["meta"]["platforms"]["llm"]["rows"] == len(LLM)
    assert r["hypotheses"]["H-budget"]["mock"]["challengers"]["BAZT"]["summary"]["n_comparisons"] > 0
    assert uncommitted == {"mock": 0, "llm": 0}


@pytest.mark.parametrize("layout", [two_workers, resumed_with_workers], ids=lambda f: f.__name__)
def test_the_report_does_not_depend_on_the_worker_count_that_wrote_the_rows(tmp_path, reference, layout):
    layout(tmp_path)
    text, uncommitted, r = report(tmp_path)
    assert r["meta"]["platforms"]["mock"]["rows"] == len(MOCK) and r["meta"]["platforms"]["llm"]["rows"] == len(LLM)
    assert text == reference[0]
    assert uncommitted == {"mock": 0, "llm": 0}


def test_rows_files_left_uncommitted_in_a_stale_shard_are_not_read_and_are_counted(tmp_path, reference):
    stale_shard_leftovers(tmp_path)
    text, uncommitted, r = report(tmp_path)
    assert text == reference[0]                                  # read once, not a DuplicateRows
    assert uncommitted == {"mock": 1, "llm": 1}
    assert f"mock: {len(MOCK)} rows, 1 uncommitted rows file not read" in cli.render(r)


def test_a_block_with_no_committed_row_is_a_block_without_data_not_an_ok_result_with_no_comparisons(tmp_path):
    stores = Stores(tmp_path)
    for k, start in enumerate(range(0, len(MOCK), BATCH)):                  # every batch cut before its done list
        stores("mock_frontier", f"w{k % 2}").append_batch({f"u{i:05d}": [synth.record(MOCK[i])]
                                                           for i in range(start, min(start + BATCH, len(MOCK)))})
    for done in (tmp_path / "mock_frontier" / "units").rglob("*.done"):
        done.unlink()
    r = analysis.analyze(tmp_path, synth.settings(), out=tmp_path / "report.json")
    mock = r["meta"]["platforms"]["mock"]
    assert mock["rows"] == 0 and mock["uncommitted_files"] == -(-len(MOCK) // BATCH)
    assert "mock_frontier" in mock["blocks_missing"]
    assert r["hypotheses"]["H-budget"]["mock"]["status"] == "no_data" and r["verdicts"]["H-budget"]["mock"] == "no_data"


def test_the_rows_of_a_unit_are_read_once_it_is_marked_done(tmp_path):
    store = core.CheckpointStore(tmp_path / "llm_capped")
    store.append_rows("u", [synth.record(r) for r in LLM[:3]])
    running = tb.load(tmp_path, ("llm_capped",), ("asr_final",))
    assert (running.n, running.uncommitted_files) == (0, 1)
    store.mark_done("u")
    done = tb.load(tmp_path, ("llm_capped",), ("asr_final",))
    assert (done.n, done.uncommitted_files) == (3, 0)
