"""The public surface, used the way the runner and the platform use it."""
import torch

from bazt_lab.features.adaptive import public as adaptive
from bazt_lab.features.adaptive.tests.fakes import FakePlatform, arm_of, make_settings, unit, units_of

N = 5


def test_the_public_names():
    assert sorted(adaptive.__all__) == sorted([
        "BestResponses", "CONDITIONS", "Choice", "Space", "TABLE_DIR", "attack_rounds", "candidate_count", "candidates",
        "load_best_responses", "load_space", "run_search", "search_episodes"])
    for name in adaptive.__all__:
        assert hasattr(adaptive, name)


def test_search_then_evaluate_the_way_the_runner_would(tmp_path):
    settings = make_settings({"mock.batch_size": 100})
    # the evaluation units of the mock_adaptive block: the attack-free units sit beside the attacked ones
    cell = unit(agents=N, condition="attack_rate", arm="C", audit=0.05)
    eval_units = units_of(cell, (1, 2), range(N)) + units_of(dict(cell, attack="none"), (1, 2), range(N))
    expected = adaptive.search_episodes(eval_units, settings)
    assert expected == adaptive.candidate_count("infection", N, "attack_rate", adaptive.load_space(settings)) * 3 * N

    platform = FakePlatform(lambda arm, u: 1.0 - abs(u["attack_params"].get("rate", 1.0) - 0.2))
    adaptive.run_search(eval_units, arm_of, platform, settings, tmp_path,
                        thresholds_of=lambda c: {"m": 1.0})
    assert platform.submitted == expected

    table = adaptive.load_best_responses(tmp_path, settings)
    assert table.missing(eval_units) == []
    annotated = [table.annotate(u) for u in eval_units]
    assert [u["attack_params"] for u in annotated if u["attack"] == "infection"] == \
        [{"entry_ranks": [0], "rate": 0.2}] * (2 * N)
    assert all(u["attack_params"] == {} for u in annotated if u["attack"] == "none")


def test_attack_rounds_is_the_schedule_the_platform_applies():
    assert adaptive.attack_rounds(10, 0.4, 0.4).tolist() == [False] * 4 + [True, False, True, False, False, True]
    assert adaptive.attack_rounds(10, 0.0, 1.0).dtype == torch.bool
