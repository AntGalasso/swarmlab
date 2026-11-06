"""Owner decisions D100-D103 and the adversary fixes: rotating sink, sanitize modes, memory-attack correctness modes,
the attack-free twin, GNN training graphs and retraining."""
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.llm_data import public as data
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.tests.test_platform import (
    SETTINGS,
    N,
    Passive,
    SanitizeAll,
    arm,
    run,
    unit,
)


@pytest.fixture()
def platform(families):
    return harness.LLMPlatform(families=lambda name: families[name])


def memory_items(k_text="Who?"):
    src = data.Source("x", None, None, None, None)
    meta = {"incorrect_answer": "Bob", "adv_texts": ["p1", "p2"], "native_id": "1"}
    return lambda t, seed, k, s: [data.TaskItem(t, f"{t}:{i}", k_text, (), "Alice", src, meta) for i in range(k)]


def test_the_sink_rotates_through_the_positions_across_conversations(platform, tmp_path):
    rows, _ = run(platform, tmp_path, conversations=3)
    assert [r["sink"] for r in rows] == [0, 1, 0]


def test_the_defender_context_follows_the_rotating_sink(platform, tmp_path):
    sinks = []

    class Look(Passive):
        def decide(self, obs):
            sinks.append(int(self.ctx.sink[0]))
            return super().decide(obs)

    run(platform, tmp_path, cls=Look, conversations=3)
    assert sinks == [0, 0, 1, 1, 0, 0]


def test_an_unknown_or_missing_sanitize_mode_is_refused(platform, tmp_path):
    with pytest.raises(ValueError, match="sanitize_mode"):
        run(platform, tmp_path, sanitize_mode="paraphrase")
    bare = unit()
    del bare["sanitize_mode"]
    with pytest.raises(ValueError, match="sanitize_mode"):
        platform.run_batch([bare], arm(Passive), SETTINGS, core.CheckpointStore(tmp_path / "b"), core.NullTracker())


def test_llm_rewrite_costs_calls_and_removes_the_infection_like_redaction(platform, tmp_path):
    open_rows, _ = run(platform, tmp_path / "o", attack="infection")
    red, _ = run(platform, tmp_path / "r", cls=SanitizeAll, attack="infection")
    rew, _ = run(platform, tmp_path / "w", cls=SanitizeAll, attack="infection", sanitize_mode="llm_rewrite")
    assert all(r["tokens_sanitizer"] == 0 for r in red) and all(r["tokens_sanitizer"] > 0 for r in rew)
    assert all(r["asr_final"] < o["asr_final"] for r, o in zip(red, open_rows, strict=True))
    assert all(r["asr_final"] < o["asr_final"] for r, o in zip(rew, open_rows, strict=True))
    assert all(r["sanitize_mode"] == "llm_rewrite" for r in rew)


def test_llm_rewrite_keeps_the_content_of_an_insiders_answer_while_redaction_removes_it(platform, tmp_path):
    rewritten, _ = run(platform, tmp_path, cls=SanitizeAll, sanitize_mode="llm_rewrite")
    redacted, _ = run(platform, tmp_path / "r", cls=SanitizeAll)
    assert all(r["asr_final"] == 1.0 for r in rewritten) and all(r["asr_final"] == 0.0 for r in redacted)


def test_memory_attack_gets_one_row_per_correctness_mode_from_the_same_conversation(families, tmp_path):
    both = {**SETTINGS, "llm": {**SETTINGS["llm"], "ma_correctness": ["rule", "llm_judge"]}}
    p = harness.LLMPlatform(families=lambda name: families[name], sample=memory_items())
    rows = p.run_batch([unit(attack="insider_ma", task="poisonedrag_msmarco")], arm(Passive), both,
                       core.CheckpointStore(tmp_path), core.NullTracker())
    assert [(r["conversation"], r["ma_check"]) for r in rows] == [(0, "rule"), (0, "llm_judge"), (1, "rule"),
                                                                  (1, "llm_judge")]
    rule, judged = rows[0], rows[1]
    assert rule["tokens_ma_judge"] == 0 and judged["tokens_ma_judge"] > 0
    assert rule["tokens_agent"] == judged["tokens_agent"] and rule["asr_final"] == judged["asr_final"] == 1.0


def test_a_memory_attack_run_resumes_without_duplicating_either_mode(families, tmp_path):
    both = {**SETTINGS, "llm": {**SETTINGS["llm"], "ma_correctness": ["rule", "llm_judge"]}}
    p = harness.LLMPlatform(families=lambda name: families[name], sample=memory_items())
    store = core.CheckpointStore(tmp_path)
    u = unit(attack="insider_ma", task="poisonedrag_msmarco")
    rows = p.run_batch([u], arm(Passive), both, store, core.NullTracker())
    other = core.CheckpointStore(tmp_path / "half")
    other.append_rows("u1", rows[:1])                          # only the first mode of conversation 0 was saved
    again = p.run_batch([u], arm(Passive), both, other, core.NullTracker())
    assert [(r["conversation"], r["ma_check"]) for r in again] == [(0, "rule"), (0, "llm_judge"), (1, "rule"),
                                                                   (1, "llm_judge")]


def test_other_units_have_one_row_with_ma_check_none(platform, tmp_path):
    rows, _ = run(platform, tmp_path)
    assert len(rows) == 2 and all(r["ma_check"] is None for r in rows)


def test_a_memory_unit_with_an_unknown_correctness_mode_is_refused(families, tmp_path):
    bad = {**SETTINGS, "llm": {**SETTINGS["llm"], "ma_correctness": ["vibes"]}}
    p = harness.LLMPlatform(families=lambda name: families[name], sample=memory_items())
    with pytest.raises(ValueError, match="ma_correctness"):
        p.run_batch([unit(attack="insider_ma", task="poisonedrag_msmarco")], arm(Passive), bad,
                    core.CheckpointStore(tmp_path), core.NullTracker())


def test_an_attack_free_twin_sees_the_same_graph_questions_and_sinks(platform, tmp_path):
    graphs, sinks = {}, {}

    class Look(Passive):
        def decide(self, obs):
            graphs.setdefault(self.tag, []).append(self.ctx.adj.clone())
            sinks.setdefault(self.tag, []).append(int(self.ctx.sink[0]))
            return super().decide(obs)

    class Attacked(Look):
        tag = "attack"

    class Free(Look):
        tag = "free"

    a, _ = run(platform, tmp_path / "a", cls=Attacked, conversations=3)
    f, _ = run(platform, tmp_path / "f", cls=Free, conversations=3, attack="none")
    assert [r["item"] for r in a] == [r["item"] for r in f]
    assert [r["sink"] for r in a] == [r["sink"] for r in f]
    assert [r["conversation"] for r in a] == [r["conversation"] for r in f] == [0, 1, 2]
    assert all(x.equal(y) for x, y in zip(graphs["attack"], graphs["free"], strict=True))
    assert sinks["attack"] == sinks["free"]


def test_judge_format_counters_are_in_every_row(platform, tmp_path):
    rows, _ = run(platform, tmp_path)
    assert all(r["meta_judge_off_format"] == 0 and r["meta_judge_missing_yes"] == 0 for r in rows)


def gnn_unit(**kw):
    return {"id": "g1", "block": "gnn_train", "platform": "llm", "arm": "GNN", "attack": "insider_pi", "task": "mmlu",
            "edge_probability": 0.6, "attackers": 2, "backbone": "fake/b", "conversations": 6, "agents": N, "rounds": 3,
            **kw}


def test_gnn_conversations_come_from_several_graphs_each_with_its_own_attackers(platform, tmp_path, monkeypatch):
    from bazt_lab.features.gnn import public as gnn
    logged = []
    monkeypatch.setattr(gnn, "log_conversations", lambda settings, u, records: logged.extend(records))
    monkeypatch.setattr(gnn, "train_unit", lambda u, s: None)
    monkeypatch.setattr(gnn, "training_finished", lambda s, f: True, raising=False)
    platform.run_batch([gnn_unit()], core.ArmSpec("GNN", Passive), SETTINGS, core.CheckpointStore(tmp_path),
                       core.NullTracker())
    graphs = {str(r["adj"]) for r in logged}
    assert 1 < len(graphs) <= 3 and len(logged) == 6
    by_graph = {}
    for r in logged:
        by_graph.setdefault(str(r["adj"]), set()).add(tuple(r["attackers"]))
    assert all(len(v) == 1 and len(next(iter(v))) == 2 for v in by_graph.values())


def test_training_is_asked_again_on_every_run_even_for_a_finished_unit(platform, tmp_path, monkeypatch):
    from bazt_lab.features.gnn import public as gnn
    asked, finished = [], [False]
    monkeypatch.setattr(gnn, "log_conversations", lambda settings, u, records: None)
    monkeypatch.setattr(gnn, "train_unit", lambda u, s: asked.append(u["id"]))
    monkeypatch.setattr(gnn, "training_finished", lambda s, f: finished[0], raising=False)
    store = core.CheckpointStore(tmp_path)
    for _ in range(2):
        platform.run_batch([gnn_unit(conversations=2)], core.ArmSpec("GNN", Passive), SETTINGS, store, core.NullTracker())
    assert asked == ["g1", "g1"] and store.is_done("g1")
    finished[0] = True
    platform.run_batch([gnn_unit(conversations=2)], core.ArmSpec("GNN", Passive), SETTINGS, store, core.NullTracker())
    assert asked == ["g1", "g1"]
