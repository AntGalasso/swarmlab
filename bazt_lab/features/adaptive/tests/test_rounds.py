"""attack_rounds: the reference definition of `start_frac` and `rate` (attack_params protocol, CONTRACT.md)."""
import pytest
import torch

from bazt_lab.features.adaptive import public as adaptive


def rounds(T, start_frac=0.0, rate=1.0):
    return [t + 1 for t, on in enumerate(adaptive.attack_rounds(T, start_frac, rate).tolist()) if on]


def test_full_rate_attacks_every_round_from_start():
    assert rounds(10) == list(range(1, 11))
    assert rounds(10, start_frac=0.4) == list(range(5, 11))        # start round = 1 + floor(0.4 * 10)


def test_returns_bool_mask_of_length_T():
    m = adaptive.attack_rounds(7, 0.0, 1.0)
    assert m.dtype == torch.bool and m.shape == (7,)


def test_first_attack_is_the_start_round_for_every_rate():
    for rate in (1.0, 0.5, 0.3, 0.1, 0.05):
        assert rounds(12, 0.25, rate)[0] == 4                       # 1 + floor(0.25 * 12)


def test_rate_spaces_attacks_evenly():
    assert rounds(10, 0.0, 0.5) == [1, 3, 5, 7, 9]
    assert rounds(10, 0.0, 0.3) == [1, 4, 7]
    assert rounds(10, 0.0, 0.1) == [1]


def test_attack_count_is_ceil_of_rate_times_opportunities():
    # decimal rates must not lose an attack to float error (0.29 * 100 = 28.999999999999996)
    assert len(rounds(100, 0.0, 0.29)) == 29
    assert len(rounds(10, 0.0, 0.7)) == 7
    assert rounds(10, 0.4, 0.4) == [5, 7, 10]                       # 6 opportunities (5..10): ceil(6 * 0.4) = 3
    assert rounds(10, 0.5, 0.4) == [6, 8]                           # 5 opportunities: ceil(5 * 0.4) = 2


def test_attack_count_never_decreases_with_rate():
    counts = [len(rounds(12, 0.0, r)) for r in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)]
    assert counts == sorted(counts)


def test_start_beyond_horizon_gives_no_attack_rounds():
    assert rounds(10, 0.99) == [10]
    assert not adaptive.attack_rounds(10, 1.0, 1.0).any()


@pytest.mark.parametrize("start_frac,rate", [(-0.1, 1.0), (1.1, 1.0), (0.0, 0.0), (0.0, 1.5)])
def test_rejects_out_of_range_arguments(start_frac, rate):
    with pytest.raises(ValueError):
        adaptive.attack_rounds(10, start_frac, rate)
