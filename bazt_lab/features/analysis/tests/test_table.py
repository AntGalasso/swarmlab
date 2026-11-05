"""analysis.table: reading the launcher's unit files, canonical codes, groups, pairing by episode."""
import math
import random

import numpy as np
import pytest

from bazt_lab.features.analysis.private import table as tb
from bazt_lab.features.analysis.tests import synth

METRICS = ("asr_final", "accuracy_clean")


def row(arm="C", seed=1, sink=0, asr=0.2, acc=math.nan, audit="uncapped", intervention="uncapped", **kw):
    base = dict(block="b", platform="mock", arm=arm, audit=audit, intervention=intervention, topology="chain",
                attack="insider_pi", agents=12, seed=seed, sink=sink, asr_final=asr, accuracy_clean=acc)
    base.update(kw)
    return base


def build(tmp_path, rows, blocks=("b",)):
    synth.write_rows(tmp_path, rows)
    return tb.load(tmp_path, blocks, METRICS)


def key(arm, audit="uncapped", intervention="uncapped", topology="chain", agents=12, condition=None, preset=None,
        cost_condition=None, attack_free=False, block="b", sanitize_mode=None, ma_check=None):
    return (block, arm, audit, intervention, topology, agents, condition, preset, cost_condition, sanitize_mode,
            ma_check, attack_free)


def values(t, rows, metric):
    return t.metrics[metric][rows]


def test_load_flattens_the_unit_and_keeps_metrics(tmp_path):
    t = build(tmp_path, [row(seed=1, sink=0, asr=0.25), row(arm="A", seed=1, sink=0, asr=0.5)])
    assert t.n == 2 and set(t.groups()) == {key("C"), key("A")}
    rc, ra = t.aligned([key("C"), key("A")])
    assert (values(t, rc, "asr_final").tolist(), values(t, ra, "asr_final").tolist()) == ([0.25], [0.5])


def test_missing_block_is_reported_not_fatal(tmp_path):
    t = build(tmp_path, [row()], blocks=("b", "nope"))
    assert t.n == 1 and t.missing_blocks == ["nope"]


def test_torn_last_line_is_skipped_and_counted(tmp_path):
    synth.write_rows(tmp_path, [row()])
    f = next((tmp_path / "b" / "units").glob("*.jsonl"))
    f.write_text(f.read_text() + '{"unit": {"arm": "C"')
    t = tb.load(tmp_path, ("b",), METRICS)
    assert t.n == 1 and t.torn_lines == 1


def test_codes_do_not_depend_on_file_or_row_order(tmp_path):
    rows = [row(seed=s, sink=k, arm=a, asr=0.1 * s + 0.01 * k) for s in range(1, 6) for k in range(3)
            for a in ("A", "C")]
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)  # noqa: S311  test data
    t1 = build(tmp_path / "one", rows)
    t2 = build(tmp_path / "two", shuffled)
    assert t1.vocab == t2.vocab
    a, b = t1.aligned([key("C"), key("A")]), t2.aligned([key("C"), key("A")])
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(values(t1, u, "asr_final"), values(t2, v, "asr_final"))


def test_pairing_is_an_inner_join_on_the_episode(tmp_path):
    rows = [row(arm="C", seed=s, asr=0.1 * s) for s in (1, 2, 3)] + [row(arm="A", seed=s, asr=0.5) for s in (2, 3, 4)]
    t = build(tmp_path, rows)
    rc, ra = t.aligned([key("C"), key("A")])
    assert values(t, rc, "asr_final").tolist() == [pytest.approx(0.2), pytest.approx(0.3)]
    assert values(t, ra, "asr_final").tolist() == [0.5, 0.5]
    assert len(set(t.codes["seed"][rc].tolist())) == 2


def test_missing_group_gives_none(tmp_path):
    assert build(tmp_path, [row()]).aligned([key("C"), key("ghost")]) is None


def test_duplicate_episode_inside_a_group_refuses_and_names_it(tmp_path):
    synth.write_rows(tmp_path, [row(seed=1, sink=0), row(seed=1, sink=0, asr=0.9)])
    with pytest.raises(tb.DuplicateRows, match="C"):
        tb.load(tmp_path, ("b",), METRICS)


def test_attack_free_rows_are_their_own_group(tmp_path):
    t = build(tmp_path, [row(), row(attack="none", acc=0.9, asr=math.nan)])
    assert set(t.groups()) == {key("C"), key("C", attack_free=True)}


def test_conflicting_attack_free_flag_refuses(tmp_path):
    synth.write_rows(tmp_path, [row(attack="none", attack_free=False)])
    with pytest.raises(ValueError, match="attack_free"):
        tb.load(tmp_path, ("b",), METRICS)


def test_aligned_joins_three_groups(tmp_path):
    rows = ([row(arm=a, seed=s) for a in ("A", "B", "C") for s in (1, 2, 3)]
            + [row(arm="A", seed=9)])                                         # unmatched by the others
    picked = build(tmp_path, rows).aligned([key("A"), key("B"), key("C")])
    assert len(picked) == 3 and all(len(p) == 3 for p in picked)


def test_missing_metric_column_is_nan_and_reported(tmp_path):
    t = build(tmp_path, [row()])
    assert np.isnan(t.metrics["accuracy_clean"]).all() and "accuracy_clean" in t.empty_metrics


def test_budget_values_are_normalised_and_order_keys_put_uncapped_last(tmp_path):
    t = build(tmp_path, [row(audit=1, intervention=0.1), row(arm="A", audit="uncapped", intervention=0.5)])
    assert key("C", 1.0, 0.1) in t.groups()
    assert tb.budget_order(0.1) < tb.budget_order(0.5) < tb.budget_order("uncapped")


def test_nan_in_json_is_read_back_as_nan(tmp_path):
    t = build(tmp_path, [row(asr=math.nan)])
    assert math.isnan(t.metrics["asr_final"][0])
    assert "NaN" in next((tmp_path / "b" / "units").glob("*.jsonl")).read_text()


def test_no_rows_at_all_is_an_empty_table(tmp_path):
    t = tb.load(tmp_path, ("b",), METRICS)
    assert t.n == 0 and t.groups() == {}


def test_load_reads_valid_json_lines_only(tmp_path):
    synth.write_rows(tmp_path, [row()])
    f = next((tmp_path / "b" / "units").glob("*.jsonl"))
    f.write_text("\n" + f.read_text() + "\n\n")
    assert tb.load(tmp_path, ("b",), METRICS).n == 1


# ---------------------------------------------------------------------- blocks and run conditions
def test_the_same_cell_in_two_blocks_is_two_groups_not_a_duplicate(tmp_path):
    rows = [row(block="b1", asr=0.1), row(block="b2", asr=0.9)]               # same arm, cell, topology, episode
    t = build(tmp_path, rows, blocks=("b1", "b2"))
    assert set(t.groups()) == {key("C", block="b1"), key("C", block="b2")}


def test_the_block_is_the_folder_the_rows_were_found_in(tmp_path):
    synth.write_rows(tmp_path, [row(block="b1")])
    (tmp_path / "b1").rename(tmp_path / "renamed")
    assert set(tb.load(tmp_path, ("renamed",), METRICS).groups()) == {key("C", block="renamed")}


def test_cost_condition_separates_groups(tmp_path):
    rows = [row(asr=0.1), row(asr=0.9, cost_condition="robustness")]
    synth.write_rows(tmp_path, rows)
    t = tb.load(tmp_path, ("b",), METRICS)
    assert set(t.groups()) == {key("C"), key("C", cost_condition="robustness")}


def test_context_of_a_key_drops_the_arm_and_the_attack_flag():
    k = tb.Key(*key("C", 0.1, 0.2, attack_free=True))
    assert k.ctx == tb.Key("b", None, 0.1, 0.2, "chain", 12, None, None, None, None, None, None)
    assert k.with_arm("A", False) == key("A", 0.1, 0.2)


def test_sort_key_orders_mixed_values_without_error():
    mixed = [tb.sort_key((None, "b", 0.5, "a", 1)), tb.sort_key((None, "a", 0.5, "a", 1))]
    assert sorted(mixed)[0] == tb.sort_key((None, "a", 0.5, "a", 1))
