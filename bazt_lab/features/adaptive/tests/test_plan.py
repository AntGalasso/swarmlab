"""The search plan: which episodes the attacker is optimised on, and which evaluation episodes share a strategy."""
import pytest

from bazt_lab.features.adaptive.private import search as se
from bazt_lab.features.adaptive.private import space as sp
from bazt_lab.features.adaptive.private import table as tb
from bazt_lab.features.adaptive.tests.fakes import make_settings, unit, units_of

N = 6
CELL = tb.cell_of(unit(agents=N))
SEARCH_SEEDS = [8001, 8002, 8003]


def plan(units, over=None):
    return se.plan_jobs(units, sp.load_space(make_settings(over)))


def eval_units(cell=None, seeds=(1, 2, 3), sinks=range(N)):
    return units_of(cell or unit(agents=N), seeds, sinks)


# ---------------------------------------------------------------------------------------- gray box: cell level
def test_a_cell_level_condition_is_searched_on_the_search_seeds_with_every_sink():
    (job,) = plan(eval_units())
    assert job.level == "cell" and job.cell == CELL
    assert (job.arm, job.family, job.agents, job.condition) == ("X", "infection", N, "gray_box")
    assert job.pairs == tuple((s, k) for s in SEARCH_SEEDS for k in range(N))
    assert job.groups == {(None, None): tuple(range(len(job.pairs)))}


def test_without_every_sink_the_sink_is_not_drawn_into_the_search():
    (job,) = plan(eval_units(sinks=[None]), {"mock.every_sink": False})
    assert job.pairs == tuple((s, None) for s in SEARCH_SEEDS)


def test_a_cell_level_search_refuses_evaluation_seeds_among_its_search_seeds():
    with pytest.raises(ValueError, match="search seeds"):
        plan(eval_units(seeds=(1, 8002)))


# --------------------------------------------------------------------------------------- white box: finer levels
def test_an_episode_level_condition_is_searched_on_the_evaluated_episodes_themselves():
    (job,) = plan(eval_units(unit(agents=N, condition="white_box"), seeds=(1, 2), sinks=(0, 3)))
    assert job.level == "episode"
    assert job.pairs == ((1, 0), (1, 3), (2, 0), (2, 3))
    assert job.groups == {(1, 0): (0,), (1, 3): (1,), (2, 0): (2,), (2, 3): (3,)}


def test_a_seed_level_condition_groups_the_sinks_of_a_seed():
    (job,) = plan(eval_units(unit(agents=N, condition="white_box"), seeds=(1, 2), sinks=(0, 3)),
                  {"mock.adaptive.knowledge.white_box": "seed"})
    assert job.level == "seed"
    assert job.pairs == ((1, 0), (1, 3), (2, 0), (2, 3))
    assert job.groups == {(1, None): (0, 1), (2, None): (2, 3)}


def test_repeated_evaluation_episodes_are_searched_once():
    u = eval_units(unit(agents=N, condition="white_box"), seeds=(1,), sinks=(0,))
    (job,) = plan(u + u)
    assert job.pairs == ((1, 0),)


# ----------------------------------------------------------------------------------------------- the cells
def test_one_job_per_cell_in_first_seen_order_and_the_attack_free_family_is_skipped():
    cells = [unit(agents=N), unit(agents=N, audit=0.2), unit(agents=N, attack="none"),
             unit(agents=N, condition="attack_rate"), unit(agents=N, attack="aitm")]
    units = [u for c in cells for u in units_of(c, (1, 2), (0, 1))]
    jobs = plan(units)
    assert [(j.cell["audit"], j.family, j.condition) for j in jobs] == [
        (0.1, "infection", "gray_box"), (0.2, "infection", "gray_box"), (0.1, "infection", "attack_rate"),
        (0.1, "aitm", "gray_box")]


def test_a_condition_outside_the_configured_ones_is_refused_by_name():
    with pytest.raises(ValueError, match="stealth"):
        plan(eval_units(unit(agents=N, condition="stealth")))


@pytest.mark.parametrize("field", ["arm", "attack", "agents", "condition"])
def test_a_unit_without_a_required_field_is_refused_by_name(field):
    u = eval_units()
    for x in u:
        del x[field]
    with pytest.raises(ValueError, match=field):
        plan(u)


# ------------------------------------------------------------------------------------------------- sizing
def test_search_episodes_is_candidates_times_search_pairs():
    space = sp.load_space(make_settings())
    units = eval_units() + eval_units(unit(agents=N, condition="embedding_evasion"))
    expected = (sp.candidate_count("infection", N, "gray_box", space)
                + sp.candidate_count("infection", N, "embedding_evasion", space)) * 3 * N
    assert se.search_episodes(units, space) == expected == (1 + N * 4) * 18 + (1 + N * 4 * 7) * 18


def test_the_candidate_cap_refuses_at_planning_before_anything_runs():
    with pytest.raises(ValueError, match="max_candidates"):
        plan(eval_units(), {"mock.adaptive.max_candidates": 10})
