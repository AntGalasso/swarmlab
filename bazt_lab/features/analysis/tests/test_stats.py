"""analysis.stats: cluster bootstrap, paired tests (Wilcoxon, McNemar exact), Holm, labels (spec section 7)."""
import numpy as np
import pytest

import stats_check
from bazt_lab.features.analysis.adapters import stats_check_adapter as sc
from bazt_lab.features.analysis.private import stats as st

R, LEVEL = 1000, 0.95


def clusters(n_seeds=5, per_seed=12):
    return np.repeat(np.arange(n_seeds), per_seed)


# ------------------------------------------------------------------------------------- cluster bootstrap
def test_cluster_ci_is_wider_than_naive_when_seeds_are_correlated():
    rng = np.random.default_rng(1)
    d = np.repeat(rng.normal(0.0, 1.0, 5), 12) + rng.normal(0.0, 0.1, 60)      # a strong seed effect
    naive = st.bootstrap_ci(d, np.arange(60), ("naive",), R, LEVEL)             # every conversation its own cluster
    cluster = st.bootstrap_ci(d, clusters(), ("cluster",), R, LEVEL)
    assert (cluster[1] - cluster[0]) > 2.5 * (naive[1] - naive[0])


def test_cluster_ci_matches_naive_when_there_is_no_seed_effect():
    rng = np.random.default_rng(2)
    d = rng.normal(0.3, 1.0, 600)
    naive = st.bootstrap_ci(d, np.arange(600), ("a",), R, LEVEL)
    cluster = st.bootstrap_ci(d, clusters(50, 12), ("b",), R, LEVEL)
    assert (cluster[1] - cluster[0]) == pytest.approx(naive[1] - naive[0], rel=0.35)
    assert naive[0] < d.mean() < naive[1]


def test_bootstrap_is_keyed_deterministic_and_independent_of_row_order():
    rng = np.random.default_rng(3)
    d, cl = rng.normal(0, 1, 60), clusters()
    a = st.bootstrap_ci(d, cl, ("k",), R, LEVEL)
    assert a == st.bootstrap_ci(d, cl, ("k",), R, LEVEL)
    assert a != st.bootstrap_ci(d, cl, ("other",), R, LEVEL)
    perm = rng.permutation(60)
    np.testing.assert_allclose(st.bootstrap_ci(d[perm], cl[perm], ("k",), R, LEVEL), a, atol=1e-12)


def test_bootstrap_ci_covers_the_true_mean_at_about_the_nominal_rate():
    rng = np.random.default_rng(4)
    hits = 0
    for rep in range(200):
        d = rng.normal(0.2, 1.0, 120)
        lo, hi = st.bootstrap_ci(d, clusters(30, 4), ("cov", rep), R, LEVEL)
        hits += lo <= 0.2 <= hi
    assert 0.88 <= hits / 200 <= 0.995


def test_bootstrap_with_one_cluster_has_no_interval():
    lo, hi = st.bootstrap_ci(np.array([0.1, 0.2, 0.3]), np.zeros(3, dtype=int), ("one",), R, LEVEL)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------------------- paired tests
def test_wilcoxon_detects_a_planted_shift_and_not_a_null():
    rng = np.random.default_rng(5)
    y = rng.normal(0.5, 0.1, 60)
    shifted = y - 0.1 + rng.normal(0, 0.05, 60)
    null = y + rng.normal(0, 0.05, 60)
    real = st.paired_test(shifted, y, clusters(), "continuous", ("real",), R, LEVEL)
    nope = st.paired_test(null, y, clusters(), "continuous", ("null",), R, LEVEL)
    assert real["p_raw"] < 1e-6 and real["delta"] == pytest.approx(-0.1, abs=0.03) and real["ci"][1] < 0
    assert real["rrb"] > 0.9                                           # x lower than y almost always
    assert nope["p_raw"] > 0.05 and nope["ci"][0] < 0 < nope["ci"][1]
    assert real["n_pairs"] == 60 and real["n_clusters"] == 5 and real["test"] == "wilcoxon"


def test_mcnemar_exact_matches_the_hand_computed_p_value():
    x = np.array([0] * 9 + [1] * 1 + [1] * 20, dtype=float)             # 9 pairs (0, 1), 1 pair (1, 0)
    y = np.array([1] * 9 + [0] * 1 + [1] * 20, dtype=float)
    r = st.paired_test(x, y, np.arange(30) % 5, "binary", ("mc",), R, LEVEL)
    assert r["test"] == "mcnemar_exact"
    assert r["p_raw"] == pytest.approx(2 * (1 + 10) / 1024)             # 2 * P(Binomial(10, 1/2) <= 1)
    assert r["delta"] == pytest.approx(-8 / 30)
    assert (r["n_discordant"], r["n_x_only"], r["n_y_only"]) == (10, 1, 9)


def test_mcnemar_without_discordant_pairs_has_p_one():
    r = st.paired_test(np.ones(10), np.ones(10), np.arange(10) % 5, "binary", ("mc0",), R, LEVEL)
    assert r["p_raw"] == 1.0 and r["delta"] == 0.0


def test_binary_metric_with_other_values_is_rejected():
    with pytest.raises(ValueError, match="binary"):
        st.paired_test(np.array([0.0, 0.5]), np.array([0.0, 1.0]), np.arange(2), "binary", ("bad",), R, LEVEL)


def test_unknown_metric_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        st.paired_test(np.ones(3), np.ones(3), np.arange(3), "ordinal", ("bad",), R, LEVEL)


def test_nan_pairs_are_dropped_and_counted():
    x = np.array([0.1, np.nan, 0.3, 0.2])
    y = np.array([0.2, 0.2, np.nan, 0.5])
    r = st.paired_test(x, y, np.arange(4) % 2, "continuous", ("nan",), R, LEVEL)
    assert r["n_pairs"] == 2 and r["n_dropped_nan"] == 2


def test_se_ratio_flags_clustering():
    rng = np.random.default_rng(6)
    d = np.repeat(rng.normal(0, 1, 5), 12) + rng.normal(0, 0.1, 60)
    r = st.paired_test(d, np.zeros(60), clusters(), "continuous", ("se",), R, LEVEL)
    assert r["se_ratio_cluster_vs_naive"] > 2.0


def test_collapse_to_cluster_means_makes_one_pair_per_seed():
    x = np.array([1.0, 3.0, 2.0, 4.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    cl = np.array([7, 7, 9, 9])
    cx, cy, cc = st.collapse(x, y, cl)
    np.testing.assert_allclose(cx, [2.0, 3.0])
    np.testing.assert_allclose(cy, [0.0, 1.0])
    assert list(cc) == [7, 9]


# ------------------------------------------------------------------------------------------- Holm, labels
def test_holm_adjusts_in_step_down_order():
    adj = sc.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


def test_holm_caps_at_one_and_handles_one_test():
    assert sc.holm({"only": 0.2}) == {"only": 0.2}
    assert sc.holm({"a": 0.9, "b": 0.8})["a"] == 1.0


@pytest.mark.parametrize("d_s, p_s, loss, p_u, ci_hi, expected", [
    (-0.2, 0.001, -0.05, 0.001, -0.01, sc.LABEL_STRICT),                 # safer and more accurate, both significant
    (-0.2, 0.001, 0.004, 0.5, 0.008, sc.LABEL_SAFETY),                   # safer, accuracy loss CI inside the margin
    (-0.2, 0.001, 0.05, 0.001, 0.07, sc.LABEL_UTILITY_LOSS),             # safer but significantly less accurate
    (-0.2, 0.001, 0.02, 0.5, 0.05, sc.LABEL_CONTAIN),                    # safer, no utility bound established
    (-0.2, 0.2, 0.0, 0.5, 0.005, sc.LABEL_NULL),                         # nothing significant
    (0.2, 0.001, 0.0, 0.5, 0.005, sc.LABEL_NULL),                        # significantly less safe is never a win
])
def test_labels_follow_the_stats_check_rules(d_s, p_s, loss, p_u, ci_hi, expected):
    got = sc.label(d_s, p_s, loss, p_u, (loss - 0.01, ci_hi), margin=0.01, alpha=0.05)
    assert got == expected
    assert got == stats_check.pareto_label(d_s, p_s, loss, p_u, (loss - 0.01, ci_hi), 0.01)


def test_label_refuses_an_alpha_stats_check_cannot_honour():
    with pytest.raises(ValueError, match="alpha"):
        sc.label(-0.2, 0.001, 0.0, 0.5, (0.0, 0.0), margin=0.01, alpha=0.01)


# ------------------------------------------------------------------------------- two independent samples
def test_unpaired_test_detects_a_planted_shift_and_not_a_null():
    rng = np.random.default_rng(7)
    a, b, null = rng.normal(1.0, 0.2, 30), rng.normal(0.5, 0.2, 30), rng.normal(0.5, 0.2, 30)
    real = st.unpaired_test(a, np.arange(30), b, np.arange(30), ("real",), R, LEVEL)
    nope = st.unpaired_test(null, np.arange(30), b, np.arange(30), ("null",), R, LEVEL)
    assert real["test"] == "mann_whitney_u" and real["p_raw"] < 1e-6
    assert real["delta"] == pytest.approx(a.mean() - b.mean()) and real["ci"][0] > 0
    assert nope["p_raw"] > 0.05 and nope["ci"][0] < 0 < nope["ci"][1]
    assert (real["n_a"], real["n_b"], real["n_clusters_a"], real["n_clusters_b"]) == (30, 30, 30, 30)


def test_unpaired_ci_resamples_each_sample_by_its_own_clusters():
    rng = np.random.default_rng(8)
    a = np.repeat(rng.normal(1.0, 1.0, 5), 12) + rng.normal(0, 0.1, 60)
    b = rng.normal(0.0, 1.0, 60)
    by_seed = st.unpaired_test(a, clusters(), b, np.arange(60), ("u",), R, LEVEL)
    naive = st.unpaired_test(a, np.arange(60), b, np.arange(60), ("u",), R, LEVEL)
    assert (by_seed["ci"][1] - by_seed["ci"][0]) > 1.5 * (naive["ci"][1] - naive["ci"][0])


def test_unpaired_test_with_one_cluster_has_no_interval_and_empty_sample_no_p():
    one = st.unpaired_test(np.array([1.0, 2.0]), np.zeros(2, dtype=int), np.array([0.0, 1.0, 2.0]), np.arange(3),
                           ("one",), R, LEVEL)
    assert np.isnan(one["ci"][0]) and one["delta"] == pytest.approx(0.5)
    empty = st.unpaired_test(np.empty(0), np.empty(0, dtype=int), np.array([1.0]), np.zeros(1, dtype=int),
                             ("e",), R, LEVEL)
    assert np.isnan(empty["p_raw"]) and np.isnan(empty["delta"])


def test_unpaired_test_drops_nan_values_and_counts_them():
    r = st.unpaired_test(np.array([1.0, np.nan, 3.0]), np.arange(3), np.array([0.0, 1.0]), np.arange(2), ("n",), R, LEVEL)
    assert (r["n_a"], r["n_b"], r["n_dropped_nan"]) == (2, 2, 1)
