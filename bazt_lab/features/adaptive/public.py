"""Public interface of bazt_lab.features.adaptive. See CONTRACT.md."""
from typing import Iterable, Mapping

from bazt_lab.features.adaptive.private.rounds import attack_rounds
from bazt_lab.features.adaptive.private.run import TABLE_DIR, load_best_responses, run_search
from bazt_lab.features.adaptive.private.search import search_episodes as _search_episodes
from bazt_lab.features.adaptive.private.space import CONDITIONS, Space, candidate_count, candidates, load_space
from bazt_lab.features.adaptive.private.table import BestResponses, Choice

__all__ = [
    "BestResponses", "CONDITIONS", "Choice", "Space", "TABLE_DIR", "attack_rounds", "candidate_count", "candidates",
    "load_best_responses", "load_space", "run_search", "search_episodes",
]


def search_episodes(units: Iterable[Mapping], settings: Mapping) -> int:
    """Episodes `run_search` simulates for these evaluation units; refuses what run_search would refuse."""
    return _search_episodes(units, load_space(settings))
