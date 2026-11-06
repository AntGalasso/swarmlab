"""Counter-based keyed RNG: Philox4x32-10 derives a 96-bit key per (episode, stream, counter); a four-stage integer hash of
(element index, key) gives each draw. All int64 lanes hold 32-bit words: exact and identical on every device and batch size."""
import math
import time

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.core.private import rng

MASK = 0xFFFFFFFF

# Published Random123 known-answer vectors (DEShawResearch/random123, tests/kat_vectors):
# (rounds, counter c0..c3, key k0..k1, expected output words).
KAT = [
    (7, (0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000), (0x5F6FB709, 0x0D893F64, 0x4F121F81, 0x4F730A48)),
    (7, (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF), (0x5207DDC2, 0x45165E59, 0x4D8EE751, 0x8C52F662)),
    (7, (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0), (0x4DFCCABA, 0x190A87F0, 0xC47362BA, 0xB6B5242A)),
    (10, (0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    (10, (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (10, (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0), (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


def _philox_reference(counter, key, rounds):
    """Philox4x32 with Python big ints (Random123 philox4x32 as published), independent of the tensor code."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(rounds):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0), p1 & MASK, ((p0 >> 32) ^ c3 ^ k1), p0 & MASK
        k0, k1 = (k0 + 0x9E3779B9) & MASK, (k1 + 0xBB67AE85) & MASK
    return c0, c1, c2, c3


def _mix_reference(p, k0, k1, k2):
    """The element hash with Python big ints and the unsigned constants (no signed-multiplier trick, no int64)."""
    def stage(x, shift, c):
        return ((x ^ (x >> shift)) * c) & MASK

    a = stage(p ^ k0, 16, 0x7FEB352D)
    b = stage(a ^ k1, 15, 0x846CA68B)
    a = stage(a ^ b ^ k2, 16, 0x9E3779B1)
    b = stage(b ^ a, 15, 0x85EBCA6B)
    hi = a ^ b
    hi ^= hi >> 15
    return hi, b ^ (b >> 15)


def _keyed_uniform_reference(key, stream, counter, index):
    """keyed_uniform for one (episode key, stream, counter, element index), from the spec, in Python ints."""
    sh = core.seed_of(stream)
    w = _philox_reference((key & MASK, (key >> 32) & MASK, sh & MASK, (sh >> 32) & MASK), (counter & MASK, counter >> 32), 10)
    hi, lo = _mix_reference(index, w[0], w[1], w[2])
    return ((hi >> 5) * 67108864 + (lo >> 6)) / 2.0**53


def _ids(n, prefix="ep"):
    return [f"{prefix}{i}" for i in range(n)]


def _keys(n=1000):
    return core.episode_keys(_ids(n))


def ks_uniform(u):
    """Kolmogorov-Smirnov distance between the sample and U(0, 1)."""
    s = torch.sort(u.reshape(-1)).values
    n = s.numel()
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return max((i / n - s).max().item(), (s - (i - 1) / n).max().item())


def corr(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm())).item()


# keyed_uniform(episode_keys(["ep0", "ep1"]), "adopt", 3, (4,)), exact multiples of 2**-53, verified against the Python reference
FROZEN_UNIFORM = [[0.13069500449299676, 0.7665146113695666, 0.5041064743506386, 0.5621797113915383],
                  [0.9074990386359084, 0.6999244657379728, 0.9416801610287526, 0.09365243566953685]]
FROZEN_NORMAL = [[0.05482044883072693, -0.5264192274658409, -1.0951468741021575],
                 [-0.6752562361110369, -2.074873851196274, 1.9830647495146048]]


# ------------------------------------------------------------------------------------------------ Philox
@pytest.mark.parametrize("rounds,counter,key,expected", KAT)
def test_philox_matches_the_random123_known_answers(rounds, counter, key, expected):
    c = [torch.tensor([v], dtype=torch.int64) for v in counter]
    k = [torch.tensor([v], dtype=torch.int64) for v in key]
    out = rng.philox4x32(*c, *k, rounds=rounds)
    assert tuple(int(o.item()) for o in out) == expected
    assert all(o.dtype == torch.int64 for o in out)


def test_philox_agrees_with_the_bigint_reference_on_random_and_edge_lanes():
    g = torch.Generator().manual_seed(7)
    edge = torch.tensor([0, 1, MASK, MASK - 1, 0x80000000, 0x7FFFFFFF, 0xFFFF, 0x10000], dtype=torch.int64)
    random_lanes = torch.randint(0, 1 << 32, (4000, 6), generator=g, dtype=torch.int64)
    edge_lanes = edge[torch.randint(0, len(edge), (4000, 6), generator=g)]
    rows = torch.where(torch.rand((4000, 6), generator=g) < 0.3, edge_lanes, random_lanes).tolist()
    t = torch.tensor(rows, dtype=torch.int64).T
    out = torch.stack(rng.philox4x32(*t)).T.tolist()
    assert out == [list(_philox_reference(row[:4], row[4:], 10)) for row in rows]


def test_philox_lanes_stay_in_32_bits_and_broadcast():
    c0 = torch.arange(5, dtype=torch.int64).reshape(1, 5)
    k = torch.tensor([[1], [2], [3]], dtype=torch.int64)
    zero = torch.zeros((), dtype=torch.int64)
    out = rng.philox4x32(c0, zero, zero, zero, k, zero)
    assert all(o.shape == (3, 5) and int(o.min()) >= 0 and int(o.max()) <= MASK for o in out)


# ------------------------------------------------------------------------------------------------ element hash
def test_the_element_hash_matches_the_bigint_reference_on_random_and_edge_lanes():
    g = torch.Generator().manual_seed(21)
    edge = torch.tensor([0, 1, MASK, MASK - 1, 0x80000000, 0x7FFFFFFF, 0xFFFF, 0x10000], dtype=torch.int64)
    random_lanes = torch.randint(0, 1 << 32, (5000, 4), generator=g, dtype=torch.int64)
    edge_lanes = edge[torch.randint(0, len(edge), (5000, 4), generator=g)]
    rows = torch.where(torch.rand((5000, 4), generator=g) < 0.3, edge_lanes, random_lanes)
    hi, lo = rng.mix(*rows.T)
    assert hi.dtype == lo.dtype == torch.int64 and int(hi.min()) >= 0 and int(hi.max()) <= MASK and int(lo.max()) <= MASK
    assert [[int(a), int(b)] for a, b in zip(hi, lo, strict=True)] == [list(_mix_reference(*row)) for row in rows.tolist()]


def test_no_multiplication_can_overflow_int64():
    """x * s with a 32-bit lane x and the signed multiplier s stays below 2**63, so no result relies on wrap-around."""
    lane_max = (1 << 32) - 1
    for _, s in rng._STAGES:
        assert abs(s) <= 1 << 31 and s % 2 == 1 and lane_max * abs(s) < 1 << 63
    for n in (rng._N0, rng._N1):
        assert 0 < n < 1 << 30 and lane_max * n < 1 << 62
    assert [s & MASK for _, s in rng._STAGES] == [0x7FEB352D, 0x846CA68B, 0x9E3779B1, 0x85EBCA6B]      # the published constants


def test_the_element_hash_avalanches_every_input_bit_into_every_output_bit():
    n = 1 << 14
    g = torch.Generator().manual_seed(11)
    args = [torch.randint(0, 1 << 32, (n,), generator=g, dtype=torch.int64) for _ in range(4)]        # index, k0, k1, k2
    base = torch.stack(rng.mix(*args))
    out_bits = torch.arange(32)
    worst = 0.0
    for which in range(4):
        for bit in range(32):
            flipped = list(args)
            flipped[which] = args[which] ^ (1 << bit)
            diff = base ^ torch.stack(rng.mix(*flipped))                                               # [2, n]
            flip_rate = ((diff.unsqueeze(-1) >> out_bits) & 1).double().mean(dim=1)                    # [2, 32]
            worst = max(worst, float((flip_rate - 0.5).abs().max()))
    assert worst < 5 * 0.5 / math.sqrt(n), worst                                                       # ideal hash: about 4 sigma


def test_keyed_uniform_equals_the_python_reference_pipeline_bit_for_bit():
    ids = ["ep0", "ep:7", "chain|1|2", "x" * 40]
    keys = core.episode_keys(ids)
    for stream_name in ("adopt", "monitor|C"):
        for counter in (0, 7, 2**40 + 5, 2**62):
            got = core.keyed_uniform(keys, stream_name, counter, (3, 11), "cpu").reshape(4, 33)
            want = torch.tensor([[_keyed_uniform_reference(k, stream_name, counter, i) for i in range(33)] for k in keys.tolist()],
                                dtype=torch.float64)
            assert torch.equal(got, want), (stream_name, counter)


def test_frozen_draws_do_not_change():
    """The streams define the experiment: a code change that moves any of these numbers changes every result."""
    keys = core.episode_keys(["ep0", "ep1"])
    assert core.keyed_uniform(keys, "adopt", 3, (4,), "cpu").tolist() == FROZEN_UNIFORM
    normals = core.keyed_normal(keys, "adopt", 3, (3,), "cpu")
    assert torch.allclose(normals, torch.tensor(FROZEN_NORMAL, dtype=torch.float64), rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------------------------ episode keys
def test_episode_keys_are_seed_of_each_id_as_one_int64_tensor():
    ids = ["a", "ep:7", "chain|1|2", "ü"]
    keys = core.episode_keys(ids)
    assert keys.dtype == torch.int64 and keys.shape == (4,)
    assert keys.tolist() == [core.seed_of(i) for i in ids]
    assert int(keys.min()) >= 0
    assert core.episode_keys([]).shape == (0,)


# ------------------------------------------------------------------------------------------------ keyed_uniform
def test_keyed_uniform_has_the_documented_shape_dtype_and_range():
    keys = _keys(6)
    u = core.keyed_uniform(keys, "adopt", 0, (3, 4), "cpu")
    assert u.shape == (6, 3, 4) and u.dtype == torch.float64 and u.device.type == "cpu"
    assert float(u.min()) >= 0.0 and float(u.max()) < 1.0
    again = core.keyed_uniform(keys, "adopt", 0, (3, 4), torch.device("cpu"))
    assert torch.equal(u, again)


def test_keyed_uniform_does_not_depend_on_batch_size_or_order():
    keys = _keys(4096)
    full = core.keyed_uniform(keys, "monitor", 3, (5,), "cpu")
    assert torch.equal(full[:1], core.keyed_uniform(keys[:1], "monitor", 3, (5,), "cpu"))
    assert torch.equal(full[1234:1235], core.keyed_uniform(keys[1234:1235], "monitor", 3, (5,), "cpu"))
    perm = torch.randperm(4096, generator=torch.Generator().manual_seed(0))
    assert torch.equal(full[perm], core.keyed_uniform(keys[perm], "monitor", 3, (5,), "cpu"))
    assert torch.equal(full[100:2000], core.keyed_uniform(keys[100:2000], "monitor", 3, (5,), "cpu"))


def test_keyed_uniform_element_depends_only_on_its_index_not_on_the_shape():
    keys = _keys(8)
    small = core.keyed_uniform(keys, "s", 0, (10,), "cpu")
    big = core.keyed_uniform(keys, "s", 0, (101,), "cpu")                 # odd size: the last block is half used
    assert torch.equal(small, big[:, :10])
    assert torch.equal(core.keyed_uniform(keys, "s", 0, (4, 5), "cpu").reshape(8, 20), core.keyed_uniform(keys, "s", 0, (20,), "cpu"))
    assert torch.equal(core.keyed_uniform(keys, "s", 0, (7,), "cpu"), core.keyed_uniform(keys, "s", 0, (8,), "cpu")[:, :7])


def test_chunking_does_not_change_the_values(monkeypatch):
    keys = _keys(5)
    want = core.keyed_uniform(keys, "chunk", 1, (1000,), "cpu")
    monkeypatch.setattr(rng, "_CHUNK_ELEMENTS", 7)
    assert torch.equal(core.keyed_uniform(keys, "chunk", 1, (1000,), "cpu"), want)
    monkeypatch.setattr(rng, "_CHUNK_ELEMENTS", 1)
    assert torch.equal(core.keyed_uniform(keys, "chunk", 1, (1000,), "cpu"), want)


def test_chunking_does_not_change_the_normals_either(monkeypatch):
    keys = _keys(4)
    want = core.keyed_normal(keys, "chunk", 1, (301,), "cpu")                 # odd size: the last pair is half used
    for chunk in (1, 3, 7, 64):
        monkeypatch.setattr(rng, "_CHUNK_ELEMENTS", chunk)
        assert torch.equal(core.keyed_normal(keys, "chunk", 1, (301,), "cpu"), want), chunk


def test_streams_counters_and_keys_give_different_draws():
    keys = _keys(50)
    base = core.keyed_uniform(keys, "a", 0, (20,), "cpu")
    assert not torch.equal(base, core.keyed_uniform(keys, "b", 0, (20,), "cpu"))
    assert not torch.equal(base, core.keyed_uniform(keys, "a", 1, (20,), "cpu"))
    assert not torch.equal(base[:, :10], base[:, 10:])
    assert not torch.equal(core.keyed_uniform(keys[:1], "a", 0, (20,), "cpu"), core.keyed_uniform(keys[1:2], "a", 0, (20,), "cpu"))
    assert not torch.equal(core.keyed_uniform(keys, "a", 2**40, (20,), "cpu"), core.keyed_uniform(keys, "a", 2**40 + 2**32, (20,), "cpu"))


def test_keys_that_differ_only_in_their_high_bits_give_different_draws():
    keys = torch.tensor([5, 5 + 2**32, 5 + 2**40, 5 + 2**62], dtype=torch.int64)
    u = core.keyed_uniform(keys, "high", 0, (8,), "cpu")
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(u[i], u[j]), (i, j)


def test_counter_may_be_a_per_episode_tensor():
    keys = _keys(6)
    counters = torch.tensor([0, 1, 2, 3, 2**33, 5], dtype=torch.int64)
    per_row = core.keyed_uniform(keys, "t", counters, (4,), "cpu")
    for b in range(6):
        assert torch.equal(per_row[b], core.keyed_uniform(keys[b:b + 1], "t", int(counters[b]), (4,), "cpu")[0])
    scalar = core.keyed_uniform(keys, "t", torch.tensor(5), (4,), "cpu")
    assert torch.equal(scalar, core.keyed_uniform(keys, "t", 5, (4,), "cpu"))


def test_empty_batch_and_empty_shape():
    keys = _keys(3)
    assert core.keyed_uniform(keys[:0], "e", 0, (4,), "cpu").shape == (0, 4)
    assert core.keyed_uniform(keys, "e", 0, (0,), "cpu").shape == (3, 0)
    assert core.keyed_uniform(keys, "e", 0, (), "cpu").shape == (3,)
    assert core.keyed_normal(keys, "e", 0, (0, 2), "cpu").shape == (3, 0, 2)


def test_input_validation_names_the_problem():
    keys = _keys(3)
    with pytest.raises(TypeError, match="keys"):
        core.keyed_uniform(keys.double(), "s", 0, (2,), "cpu")
    with pytest.raises(ValueError, match="keys"):
        core.keyed_uniform(keys.reshape(3, 1), "s", 0, (2,), "cpu")
    with pytest.raises(TypeError, match="stream"):
        core.keyed_uniform(keys, 5, 0, (2,), "cpu")
    with pytest.raises(ValueError, match="counter"):
        core.keyed_uniform(keys, "s", -1, (2,), "cpu")
    with pytest.raises(ValueError, match="counter"):
        core.keyed_uniform(keys, "s", 2**63, (2,), "cpu")
    with pytest.raises(ValueError, match="shape"):
        core.keyed_uniform(keys, "s", 0, (2, -1), "cpu")
    with pytest.raises(ValueError, match="counter"):
        core.keyed_uniform(keys, "s", torch.arange(5), (2,), "cpu")      # tensor counter of the wrong length
    with pytest.raises(ValueError, match="dtype"):
        core.keyed_uniform(keys, "s", 0, (2,), "cpu", dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        core.keyed_uniform(keys, "s", 0, (2,), "cpu", dtype=torch.int64)


def test_weak_sequential_keys_still_give_independent_streams():
    keys = torch.arange(1000, dtype=torch.int64)                          # not hashed: 0, 1, 2, ...
    u = core.keyed_uniform(keys, "weak", 0, (1000,), "cpu")
    bound = 5 / math.sqrt(1000 * 999)
    assert abs(corr(u[:-1], u[1:])) < bound * 5
    assert abs(float(u.mean()) - 0.5) < 5 * math.sqrt(1 / 12 / u.numel())


# ------------------------------------------------------------------------------------------------ statistics
def test_uniform_statistics_over_a_million_draws():
    u = core.keyed_uniform(_keys(1000), "stats", 0, (1000,), "cpu")
    n = u.numel()
    assert n == 1_000_000
    assert abs(float(u.mean()) - 0.5) < 5 * math.sqrt(1 / 12 / n)
    assert abs(float(u.var()) - 1 / 12) < 5 * math.sqrt((1 / 80 - 1 / 144) / n)
    assert ks_uniform(u) < 1.95 / math.sqrt(n)                           # KS critical value at alpha = 0.001


def test_uniform_deciles_are_flat_and_all_53_bits_are_used():
    u = core.keyed_uniform(_keys(200), "bits", 4, (500,), "cpu").reshape(-1)
    counts = torch.histc(u, bins=10, min=0.0, max=1.0)
    chi2 = float(((counts - u.numel() / 10) ** 2 / (u.numel() / 10)).sum())
    assert chi2 < 27.88                                                   # chi-square, 9 dof, alpha = 0.001
    mantissa = (u * 2.0**53).to(torch.int64)
    for bit in (0, 1, 7, 26, 27, 40, 52):
        frac = float(((mantissa >> bit) & 1).double().mean())
        assert abs(frac - 0.5) < 5 * 0.5 / math.sqrt(u.numel()), bit


def test_streams_counters_and_neighbouring_episodes_are_uncorrelated():
    keys = _keys(1000)
    a = core.keyed_uniform(keys, "alpha", 0, (1000,), "cpu")
    n = a.numel()
    bound = 5 / math.sqrt(n)
    assert abs(corr(a, core.keyed_uniform(keys, "beta", 0, (1000,), "cpu"))) < bound
    assert abs(corr(a, core.keyed_uniform(keys, "alpha", 1, (1000,), "cpu"))) < bound
    assert abs(corr(a, core.keyed_uniform(keys, "alpha", 2**32, (1000,), "cpu"))) < bound
    assert abs(corr(a[:-1], a[1:])) < bound * 2                          # adjacent episodes
    assert abs(corr(a[:, :-1], a[:, 1:])) < bound                        # adjacent elements
    assert abs(corr(a[:, ::2], a[:, 1::2])) < bound * 2                  # elements 2k and 2k + 1, the Box-Muller pair


def test_normal_statistics_over_a_million_draws():
    z = core.keyed_normal(_keys(1000), "z", 0, (1000,), "cpu")
    n = z.numel()
    assert z.dtype == torch.float64 and torch.isfinite(z).all()
    assert abs(float(z.mean())) < 5 / math.sqrt(n)
    assert abs(float(z.var()) - 1) < 5 * math.sqrt(2 / n)
    centred = z - z.mean()
    assert abs(float((centred**3).mean() / z.var() ** 1.5)) < 5 * math.sqrt(6 / n)
    assert abs(float((centred**4).mean() / z.var() ** 2) - 3) < 5 * math.sqrt(24 / n)
    cdf = 0.5 * (1 + torch.special.erf(z / math.sqrt(2)))
    assert ks_uniform(cdf) < 1.95 / math.sqrt(n)
    assert abs(corr(z[:, ::2], z[:, 1::2])) < 5 / math.sqrt(n / 2)       # cos and sin halves of one Box-Muller pair


def test_normal_element_depends_only_on_its_index_and_not_on_the_batch():
    keys = _keys(300)
    full = core.keyed_normal(keys, "z", 2, (9,), "cpu")
    assert torch.equal(full[7:8], core.keyed_normal(keys[7:8], "z", 2, (9,), "cpu"))
    assert torch.equal(core.keyed_normal(keys, "z", 2, (5,), "cpu"), full[:, :5])


# ------------------------------------------------------------------------------------------------ randint / scores
def test_randint_is_in_range_int64_flat_and_equal_to_the_floor_of_the_uniform():
    keys = _keys(700)
    r = core.keyed_randint(keys, "pick", 0, (1000,), 7, "cpu")
    assert r.dtype == torch.int64 and r.shape == (700, 1000)
    assert int(r.min()) == 0 and int(r.max()) == 6
    counts = torch.bincount(r.reshape(-1), minlength=7).double()
    expected = r.numel() / 7
    assert float(((counts - expected) ** 2 / expected).sum()) < 22.46    # chi-square, 6 dof, alpha = 0.001
    u = core.keyed_uniform(keys, "pick", 0, (1000,), "cpu")
    assert torch.equal(r, (u * 7).floor().to(torch.int64))
    assert int(core.keyed_randint(keys, "pick", 0, (4,), 1, "cpu").abs().sum()) == 0


def test_randint_rejects_a_bad_bound():
    keys = _keys(2)
    for bad in (0, -3, 2**31 + 1):
        with pytest.raises(ValueError, match="high"):
            core.keyed_randint(keys, "pick", 0, (2,), bad, "cpu")


def test_permutation_scores_rank_uniformly_and_break_ties_stably():
    keys = _keys(60000)
    scores = core.keyed_permutation_scores(keys, "perm", 0, (6,), "cpu")
    assert scores.dtype == torch.float64 and scores.shape == (60000, 6)
    order = torch.argsort(scores, dim=-1, stable=True)
    assert torch.equal(torch.sort(order, dim=-1).values, torch.arange(6).expand(60000, 6))      # a permutation per row
    rank_of_first = torch.argsort(order, dim=-1)[:, 0]
    counts = torch.bincount(rank_of_first, minlength=6).double()
    expected = 60000 / 6
    assert float(((counts - expected) ** 2 / expected).sum()) < 20.52   # chi-square, 5 dof, alpha = 0.001
    assert torch.equal(scores, core.keyed_uniform(keys, "perm", 0, (6,), "cpu"))                 # same numbers, named for intent


def test_permutation_scores_stay_float64_whatever_the_compute_dtype():
    assert core.keyed_permutation_scores(_keys(2), "perm", 0, (3,), "cpu").dtype == torch.float64


# ------------------------------------------------------------------------------------------------ dtype
def test_float32_draws_are_the_float64_draws_rounded_and_still_below_one():
    keys = _keys(500)
    u64 = core.keyed_uniform(keys, "d", 0, (400,), "cpu")
    u32 = core.keyed_uniform(keys, "d", 0, (400,), "cpu", dtype=torch.float32)
    assert u32.dtype == torch.float32
    assert torch.equal(u32, u64.to(torch.float32).clamp(max=1.0 - 2.0**-24))
    assert float(u32.max()) < 1.0
    z64 = core.keyed_normal(keys, "d", 0, (400,), "cpu")
    z32 = core.keyed_normal(keys, "d", 0, (400,), "cpu", dtype=torch.float32)
    assert z32.dtype == torch.float32 and torch.equal(z32, z64.to(torch.float32))


def test_rounding_never_produces_exactly_one():
    top = torch.tensor([1.0 - 2.0**-53, 1.0 - 2.0**-30, 0.5], dtype=torch.float64)
    out = rng.as_uniform_dtype(top, torch.float32)
    assert out.dtype == torch.float32 and float(out.max()) < 1.0 and float(out[2]) == 0.5
    assert rng.as_uniform_dtype(top, torch.float64) is top


# ------------------------------------------------------------------------------------------------ devices and speed
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this machine")
def test_cuda_draws_are_identical_to_cpu():
    keys = _keys(300)
    for dev in ("cuda", "cuda:0"):
        assert torch.equal(core.keyed_uniform(keys, "dev", 5, (50, 3), "cpu"), core.keyed_uniform(keys, "dev", 5, (50, 3), dev).cpu())
        assert torch.equal(core.keyed_randint(keys, "dev", 5, (50,), 9, "cpu"), core.keyed_randint(keys, "dev", 5, (50,), 9, dev).cpu())
        assert torch.equal(core.keyed_permutation_scores(keys, "dev", 5, (8,), "cpu"), core.keyed_permutation_scores(keys, "dev", 5, (8,), dev).cpu())
        # Box-Muller uses log/sqrt/cos/sin whose last bit may differ between libm and CUDA
        assert torch.allclose(core.keyed_normal(keys, "dev", 5, (50,), "cpu"), core.keyed_normal(keys, "dev", 5, (50,), dev).cpu(), rtol=1e-12, atol=1e-12)
    cuda_keys = keys.to("cuda")
    assert core.keyed_uniform(cuda_keys, "dev", 5, (4,), "cuda").device.type == "cuda"


def test_ten_million_draws_are_a_few_thousand_vectorised_tensor_ops():
    """Load-independent speed guard: the cost is tens of thousands of elements per op, never a Python loop over draws."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountOps(TorchDispatchMode):
        count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            CountOps.count += 1
            return func(*args, **(kwargs or {}))

    keys = _keys(1000)
    with CountOps():
        out = core.keyed_uniform(keys, "ops", 0, (10_000,), "cpu")
    assert out.numel() == 10_000_000
    assert CountOps.count < 5_000, f"{CountOps.count} tensor ops for 1e7 draws"           # measured: about 3100


def test_ten_million_draws_on_cpu_take_well_under_a_second():
    keys = _keys(1000)
    core.keyed_uniform(keys, "speed", 0, (10,), "cpu")                    # warm up the thread pool and allocator
    best = min(_timed(lambda: core.keyed_uniform(keys, "speed", 0, (10_000,), "cpu")) for _ in range(5))
    if best < 0.5:
        return
    # Slow wall clock: a busy shared machine or slow code? CPU seconds on one thread do not grow as much when other
    # processes share the cores, so they tell the two apart (measured on a busy machine: 0.6 s of one-thread CPU time).
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        start = time.process_time()
        core.keyed_uniform(keys, "speed", 0, (10_000,), "cpu")
        cpu = time.process_time() - start
    finally:
        torch.set_num_threads(threads)
    if cpu < 1.0:
        pytest.skip(f"machine busy: best wall time {best:.2f} s for 1e7 draws, but only {cpu:.2f} s of one-thread CPU time")
    pytest.fail(f"1e7 draws took {best:.2f} s wall and {cpu:.2f} s of one-thread CPU time")


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


# ------------------------------------------------------------------------------------------------ old API
def test_the_generator_api_still_works_next_to_the_keyed_one():
    a = core.uniforms(core.stream("adopt", "chain", 1), (3, 4))
    assert a.dtype == torch.float64 and torch.equal(a, core.batch_uniforms([("adopt", "chain", 1)], (3, 4))[0])
