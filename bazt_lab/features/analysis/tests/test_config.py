"""analysis.config: every experiment value comes from settings; a missing or invalid one refuses with its key."""
import pytest

import stats_check
from bazt_lab.core import public as core
from bazt_lab.features.analysis.private import config
from bazt_lab.features.analysis.tests import synth


def test_all_keys_resolve_into_a_config():
    cfg = config.Cfg.from_settings(synth.settings())
    assert (cfg.alpha, cfg.resamples, cfg.level, cfg.ni_margin) == (0.05, 1000, 0.95, 0.01)
    assert cfg.baseline == "C" and cfg.challengers == ("BAZT",) and cfg.profile == "test"
    assert cfg.extra_metrics == {"protected_sink_harm": ("binary", "attacked")}
    assert cfg.blocks["budget"]["mock"] == ("mock_frontier",) and cfg.pair_unit["llm"] == "episode"
    assert cfg.utility_column == "accuracy_clean" and cfg.utility_attack_free is True
    assert cfg.contrast_factor == 2.0


def test_the_utility_endpoint_decides_which_rows_it_is_read_from():
    cfg = config.Cfg.from_settings(synth.settings(analysis__utility_endpoint="accuracy_attack"))
    assert cfg.utility_column == "accuracy_attack" and cfg.utility_attack_free is False


def test_the_adaptive_contrast_option_sets_the_factor_on_the_baselines_loss():
    cfg = config.Cfg.from_settings(synth.settings(analysis__adaptive_contrast="challenger_degrades_less_than_c"))
    assert cfg.contrast_factor == 1.0


@pytest.mark.parametrize("key", config.REQUIRED_KEYS)
def test_every_missing_key_refuses_with_its_name(key):
    tree = synth.settings()
    node = tree
    *path, leaf = key.split(".")
    for part in path:
        node = node[part]
    del node[leaf]
    with pytest.raises(core.Unset) as e:
        config.Cfg.from_settings(tree)
    assert e.value.args[0] == key
    assert config.missing_keys(tree) == [key]


def test_missing_keys_lists_everything_at_once():
    assert config.missing_keys({}) == list(config.REQUIRED_KEYS)


@pytest.mark.parametrize("key, value, text", [
    ("analysis.pair_unit.mock", "sink", "pair_unit"),
    ("analysis.utility_endpoint", "accuracy", "utility_endpoint"),
    ("analysis.adaptive_contrast", "bigger", "adaptive_contrast"),
    ("analysis.ci_level", 1.5, "ci_level"),
    ("analysis.bootstrap_resamples", 0, "bootstrap_resamples"),
    ("analysis.rank_agreement_statistic", "pearson", "rank_agreement_statistic"),
    ("analysis.extra_metrics", {"x": ["ordinal", "attacked"]}, "extra_metrics"),
    ("analysis.extra_metrics", {"x": ["binary", "sometimes"]}, "extra_metrics"),
    ("analysis.challengers", [], "challengers"),
    ("analysis.challengers", ["C"], "challengers"),                  # the baseline cannot challenge itself
    ("analysis.alpha", 0.0, "alpha"),
    ("analysis.alpha", 0.01, "alpha"),                              # stats_check.pareto_label is fixed at 0.05
    ("analysis.split_by.mock", ["sink"], "split_by"),
    ("analysis.topology_classes.mock.homogeneous", ["star_hub"], "topology_classes"),   # also in heterogeneous
    ("analysis.blocks.budget.mock", "mock_frontier", "blocks"),
    ("analysis.containment_metrics", [], "containment_metrics"),
    ("analysis.rank_agreement_arms", [["A", "A"]], "rank_agreement_arms"),
    ("runtime.ni_margin_accuracy", -0.01, "ni_margin_accuracy"),
    ("runtime.rank_agreement_threshold", 1.5, "rank_agreement_threshold"),
])
def test_invalid_values_are_rejected(key, value, text):
    with pytest.raises(ValueError, match=text):
        config.Cfg.from_settings(synth.settings(**{key.replace(".", "__"): value}))


def test_the_significance_level_is_the_one_stats_check_labels_with():
    assert config.Cfg.from_settings(synth.settings(analysis__alpha=stats_check.ALPHA)).alpha == stats_check.ALPHA


def test_metric_columns_cover_every_column_the_analysis_reads():
    cfg = config.Cfg.from_settings(synth.settings())
    assert set(cfg.metric_columns) == {"asr_final", "accuracy_clean", "spend_audit_total", "spend_intervention_total",
                                       "protected_sink_harm", "agents_reached_auc", "sink_is_attacker"}
