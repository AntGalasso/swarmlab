"""D116: `intervened_by_round`, the rounds at which the platform applied an action of the arm other than an audit, from
`meta["intervened"]` (bool [B, T], index r = round r+1). A matched-timing control reads BAZT's value as its schedule."""
import json

import pytest
import torch

from bazt_lab.features.metrics import public as metrics
from bazt_lab.features.metrics.tests.helpers import KINDS, make, one, rows

DENY, INSPECT = KINDS.index("deny"), KINDS.index("inspect")


def acted(B, T, rounds, counts=None):
    """meta["intervened"] with the given (b, r) set, and action counts that agree with it."""
    m = torch.zeros(B, T, dtype=torch.bool)
    for b, r in rounds:
        m[b, r] = True
    c = torch.zeros(B, len(KINDS), dtype=torch.int64)
    c[:, DENY] = m.sum(1) if counts is None else torch.tensor(counts)
    return m, c


def test_the_column_is_the_list_of_t_booleans_of_the_episode():
    m, c = acted(2, 3, [(0, 1), (1, 0), (1, 2)])
    a, b = rows(make(B=2, T=3, action_counts=c, meta={"intervened": m}))
    assert a["intervened_by_round"] == [False, True, False] and b["intervened_by_round"] == [True, False, True]
    assert all(type(x) is bool for x in a["intervened_by_round"])
    json.dumps(a)
    assert "intervened_by_round" in metrics.COLUMNS and "meta_intervened" not in a


def test_without_the_record_the_column_is_none():
    assert one(make())["intervened_by_round"] is None


def test_audits_alone_are_no_intervention_and_agree_with_an_empty_record():
    m, c = acted(1, 3, [])
    c[:, INSPECT] = 4
    assert one(make(action_counts=c, meta={"intervened": m}))["intervened_by_round"] == [False] * 3


@pytest.mark.parametrize("bad", [torch.zeros(1, 3), torch.zeros(1, 2, dtype=torch.bool), [[False, True, False]],
                                 torch.zeros(2, 3, dtype=torch.bool)])
def test_a_malformed_record_is_refused(bad):
    with pytest.raises(ValueError, match="intervened"):
        rows(make(meta={"intervened": bad}))


@pytest.mark.parametrize("rounds,counts", [([(0, 1)], [0]), ([], [2])])
def test_a_record_that_disagrees_with_the_action_counts_is_refused(rounds, counts):
    """An episode intervened at some round exactly when it applied some action other than an audit."""
    m, c = acted(1, 3, rounds, counts)
    with pytest.raises(ValueError, match="intervened"):
        rows(make(action_counts=c, meta={"intervened": m}))


def test_the_column_does_not_depend_on_the_batch_size():
    m, c = acted(4, 5, [(0, 0), (1, 4), (2, 1), (2, 2), (3, 3)])
    whole = rows(make(B=4, T=5, action_counts=c, meta={"intervened": m}))
    for b in range(4):
        single = one(make(B=1, T=5, action_counts=c[b:b + 1], meta={"intervened": m[b:b + 1]}))
        assert single["intervened_by_round"] == whole[b]["intervened_by_round"]
