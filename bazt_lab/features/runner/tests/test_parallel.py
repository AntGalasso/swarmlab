"""Twin-first order, worker processes, per-arm cost estimates."""
import multiprocessing
import time

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_launch import MOCK_A, block_dir, launch

BASE = [fakes.RUNTIME_MOCK, fakes.TINY_MOCK, ["mock.uncapped_arms=['A']", "mock.capped_arms=[]",
                                              "mock.allocation_arms=[]", "mock.main_seeds=[1, 4]"]]


def rows_of(out):
    return sorted(line for f in (out / "units").rglob("*.jsonl") for line in f.read_text().splitlines())


def launched(tmp_path, workers, extra=(), platform=None):
    s = fakes.settings(*BASE, [f"compute.workers={workers}", *extra])
    plat = platform or fakes.FakePlatform("mock")
    rx.run_block(s, "mock_frontier", False, wiring=fakes.wiring({"mock": plat}, rebuild=fakes.REBUILD),
                 results_root=tmp_path)
    return plat


# ================================================================================================ twin first
def test_attack_free_units_come_first_and_attacked_units_name_their_twin():
    grid = list(rx.units_mock_frontier(fakes.settings(*BASE), False))
    attacks = [u["attack"] for u in grid]
    assert attacks == sorted(attacks, key=lambda a: a != "none")                  # all "none" before any attacked
    ids = {rx.unit_id(u) for u in grid if u["attack"] == "none"}
    for u in grid:
        if u["attack"] != "none":
            twin = {**u, "attack": "none"}
            assert rx.unit_id(twin) in ids


def test_the_platform_gets_twin_id_on_attacked_units_only_and_the_id_ignores_it(tmp_path):
    plat = launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path)
    sent = [u for _, us in plat.batches for u in us]
    none, attacked = [u for u in sent if u["attack"] == "none"], [u for u in sent if u["attack"] != "none"]
    assert none and attacked and all("twin_id" not in u for u in none)
    assert {u["twin_id"] for u in attacked} <= {u["id"] for u in none}
    assert all(u["id"] == rx.unit_id({k: v for k, v in u.items() if k != "twin_id"}) for u in attacked)
    assert sent.index(none[-1]) < sent.index(attacked[0])


# ================================================================================================== workers
def test_one_and_four_workers_write_the_same_rows(tmp_path):
    launched(tmp_path / "one", 1)
    launched(tmp_path / "four", 4)
    one, four = rows_of(block_dir(tmp_path / "one", "mock_frontier")), rows_of(block_dir(tmp_path / "four", "mock_frontier"))
    assert one and one == four
    shards = {p.parent.name for p in (block_dir(tmp_path / "four", "mock_frontier") / "units").rglob("*.jsonl")}
    assert shards <= {"w0", "w1", "w2", "w3"}                           # workers write only their own folders


def test_a_worker_crash_stops_everything_and_a_later_launch_with_any_worker_count_finishes(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_SEED", "3")
    with pytest.raises(RuntimeError, match=r"worker.*simulated worker crash"):
        launched(tmp_path, 2)
    assert multiprocessing.active_children() == []
    monkeypatch.delenv("FAKE_CRASH_SEED")
    plat = launched(tmp_path, 1)                                                  # resume inline
    out = block_dir(tmp_path, "mock_frontier")
    store = core.CheckpointStore(out)
    ids = {rx.unit_id(u) for u in rx.units_mock_frontier(fakes.settings(*BASE), False)}
    assert all(store.is_done(i) for i in ids)
    assert {u["id"] for _, us in plat.batches for u in us}.isdisjoint(
        {i for i in ids if i not in {u["id"] for _, us in plat.batches for u in us}})
    rows = rows_of(out)
    assert len(rows) == len(set(rows)) == len(ids)                                # every unit exactly once


def test_a_wiring_that_cannot_be_rebuilt_in_a_worker_is_refused_before_anything_is_written(tmp_path):
    s = fakes.settings(*BASE, ["compute.workers=2"])
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: .*workers"):
        rx.run_block(s, "mock_frontier", False, wiring=fakes.wiring({"mock": fakes.FakePlatform("mock")}),
                     results_root=tmp_path)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", "'two'", "true"])
def test_compute_workers_must_be_a_positive_integer(tmp_path, bad):
    s = fakes.settings(*BASE, [f"compute.workers={bad}"])
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: compute\.workers"):
        launch(s, "mock_frontier", tmp_path)
    assert not any(tmp_path.iterdir())


def test_workers_are_not_part_of_the_identity_and_the_flag_sets_the_key(tmp_path):
    launched(tmp_path, 1)
    assert launched(tmp_path, 2).batches == []                                    # nothing left, no identity refusal
    seen = []

    class Spy(fakes.FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            seen.append(core.get(settings, "compute.workers"))
            return super().run_batch(units, arm, settings, store, tracker)

    argv = ["--run", "--block", "mock_frontier", "--workers", "1", "--results-root", str(tmp_path / "flag")]
    argv += [x for g in BASE for o in g for x in ("--set", o)]
    rx.main(argv, wiring=fakes.wiring({"mock": Spy("mock")}))
    assert set(seen) == {1}
    with pytest.raises(SystemExit):
        rx.main(["--run", "--block", "mock_frontier", "--workers", "0"], wiring=fakes.wiring())



class LateDoneLists:
    """The parent's store while another process still holds the done lists the workers committed last (on Windows the
    rename that created a done list holds it for a moment): for its next `late` calls refresh() cannot read the last one
    and reports it skipped, and its units are not counted done yet."""

    def __init__(self, store, units):
        self.store, self.units, self.late, self.holding = store, units, 0, False

    def refresh(self):
        skipped = self.store.refresh()
        self.holding = self.late > 0
        self.late -= self.holding
        return skipped + self.holding

    def done_count(self):
        return self.store.done_count() - (self.units if self.holding else 0)


def run_workers(tmp_path, monkeypatch, late):
    """workers.run on two real worker processes; the parent's store reads the last done list `late` calls late."""
    from bazt_lab.features.runner.adapters import workers
    s = fakes.settings(*BASE)
    grid = [{**u, "id": rx.unit_id(u)} for u in rx.units_mock_frontier(s, False) if u["attack"] == "none"]
    batches = [grid[k:k + 2] for k in range(0, len(grid), 2)]
    store = LateDoneLists(core.CheckpointStore(tmp_path), len(batches[-1]))
    finish = workers._Pool.finish

    def finished_then_late(self, settle):
        finish(self, settle)
        store.late = late
    monkeypatch.setattr(workers._Pool, "finish", finished_then_late)
    return workers.run(batches, 2, fakes.REBUILD, s, "mock", tmp_path, store, lambda *a: None), len(grid)


def test_the_final_count_waits_for_done_lists_it_could_not_read_yet(tmp_path, monkeypatch):
    dispatched, units = run_workers(tmp_path, monkeypatch, late=3)
    assert dispatched == units > 0


def test_the_wait_for_unreadable_done_lists_is_bounded(tmp_path, monkeypatch):
    from bazt_lab.features.runner.adapters import workers
    monkeypatch.setattr(workers, "SETTLE_SECONDS", 0.3)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match=r"units not done"):
        run_workers(tmp_path, monkeypatch, late=10 ** 9)
    assert time.monotonic() - start < 60


# ================================================================================================ estimates
def test_the_dry_run_sizes_each_arm_by_its_measured_cost_and_divides_by_the_workers(capsys):
    sized = ["mock.frontier_agents=[12]", "mock.main_seeds=[1, 30]", "estimates.mock_checkpoint_seconds_per_unit=10"]
    argv = ["--dry-run", "--block", "mock_frontier"] + [x for g in [*BASE, sized] for o in g for x in ("--set", o)]
    rx.main(argv + ["--set", "compute.workers=1"], wiring=fakes.wiring())
    one = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("mock_frontier"))
    rx.main(argv + ["--set", "compute.workers=4"], wiring=fakes.wiring())
    four = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("mock_frontier"))
    s = fakes.settings(*BASE, sized)
    table = core.get(s, "estimates.mock_seconds_per_episode_by_arm")
    n = len(rx.units_mock_frontier(s, False))
    cpu = n * (table["12"]["A|uncapped"] + 10) / 3600
    assert f"{cpu:.2f}" in one and f"{cpu / 4:.2f}" in four


def test_an_arm_without_a_measurement_takes_the_slowest_arms_cost():
    from bazt_lab.features.runner.private import sizing
    s = fakes.settings()
    table = core.get(s, "estimates.mock_seconds_per_episode_by_arm")
    slowest = max(v for k, v in table["12"].items() if k.endswith("|capped"))
    assert sizing.seconds_per_episode(s, "legacy_unmeasured", "capped", 12) == slowest
    assert sizing.seconds_per_episode(s, "BAZT", "capped", 12) == table["12"]["BAZT|capped"]
    assert sizing.seconds_per_episode(s, "BAZT", "capped", 100) == table["100"]["BAZT|capped"]
