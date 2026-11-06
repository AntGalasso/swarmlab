"""The patched G-Safeguard modules: defender hook, per-edge delivery, sender outbox, external context."""
import asyncio

import numpy as np
import pytest

REPLY = "<REASON>: because of {tag}\n<ANSWER>: {ans}"


class Plan:
    """A delivery plan: views[(src, dst)] is what dst receives from src; anything missing is not delivered."""

    def __init__(self, views):
        self.views = views

    def view(self, src, dst):
        return self.views.get((src, dst))


class Recorder:
    """Stands in for the LLM: records every prompt, answers with a fixed well-formed reply."""

    def __init__(self, reply=None):
        self.reply, self.calls = reply or REPLY.format(tag="x", ans="A"), []

    async def __call__(self, messages, model_type):
        self.calls.append([dict(m) for m in messages])
        return self.reply


def graph(fam, adj, attackers=(), system="sys"):
    ag = fam.agents.AgentGraphWithDefense(np.array(adj), [f"{system}{i}" for i in range(len(adj))], list(attackers),
                                          model_type="m")
    return ag


def run(fam, ag, hook, qa=("Q?\nA. x\nB. y", "A", "B"), turns=2):
    return asyncio.run(fam.main.defense_communication(ag, hook, qa, np.array(ag.adj_matrix), turns))


async def deliver_all(turn, responses):
    return None


def last_user(ag, idx):
    return [m for m in ag.agents[idx].memory if m["role"] == "user"][-1]["content"]


CASES = {"pi": ("Q?\nA. x\nB. y", "A", "B"), "pi_gsm8k": ("Q?", "7", None),
         "ta": ("do it", ["T"], "desc", "pad")}


def drive(fam, family, ag, hook, turns):
    """defense_communication with the argument shape of the family."""
    adj = np.array(ag.adj_matrix)
    if family == "ma":
        return asyncio.run(fam.main.defense_communication(ag, hook, "Q?", "ctx", adj, turns))
    return asyncio.run(fam.main.defense_communication(ag, hook, CASES[family], adj, turns))


def reply_for(family):
    return "Thought: t\nFinal Answer: ok" if family == "ta" else REPLY.format(tag="own", ans="A")


FULL3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]


def test_hook_is_awaited_before_every_update_with_the_responses_just_produced(families):
    fam, seen = families["pi"], []
    fam.agents.allm_invoke = Recorder()

    async def hook(turn, responses):
        seen.append((turn, sorted(i for i, _ in responses)))
        return None

    data = run(fam, graph(fam, FULL3), hook, turns=3)
    assert seen == [(0, [0, 1, 2]), (1, [0, 1, 2]), (2, [0, 1, 2])]
    assert len(data) == 4                                           # 1 initial round + 3 updates


@pytest.mark.parametrize("family", ["pi", "pi_gsm8k", "ma"])
def test_without_a_plan_every_normal_sender_reaches_every_out_neighbour(families, family):
    fam = families[family]
    fam.agents.allm_invoke = Recorder()
    ag = graph(fam, FULL3)
    drive(fam, family, ag, deliver_all, turns=1)
    prompt = last_user(ag, 0)
    assert "Agent_1's View:" in prompt and "Agent_2's View:" in prompt


def test_a_delivered_view_is_exactly_what_the_plan_returns_and_denied_edges_are_absent(families):
    fam = families["pi"]
    fam.agents.allm_invoke = Recorder()
    view = {"answer": "B", "reason": "sanitised"}

    async def hook(turn, responses):
        return Plan({(1, 0): view})                                  # only 1 -> 0 reaches anyone

    ag = graph(fam, FULL3)
    run(fam, ag, hook, turns=1)
    to0, to1 = last_user(ag, 0), last_user(ag, 1)
    assert "Agent_1's View:" in to0 and "sanitised" in to0 and "Agent_2's View:" not in to0
    assert "No responses from other agents." in to1


def test_a_graph_where_nothing_is_delivered_says_so(families):
    fam = families["ta"]
    fam.agents.allm_invoke = Recorder("Thought: nothing\nFinal Answer: ok")

    async def hook(turn, responses):
        return Plan({})

    ag = fam.agents.AgentGraphWithDefense(np.array(FULL3), ["s"] * 3, [], model_type="m")
    asyncio.run(fam.main.defense_communication(ag, hook, ("do it", ["T"], "desc", "pad"), np.array(FULL3), 1))
    assert "No responses from other agents." in last_user(ag, 0)


def test_ta_views_are_the_raw_replies(families):
    fam = families["ta"]
    fam.agents.allm_invoke = Recorder("Thought: t\nFinal Answer: ok")

    async def hook(turn, responses):
        return Plan({(2, 0): "sanitised reply"})

    ag = fam.agents.AgentGraphWithDefense(np.array(FULL3), ["s"] * 3, [], model_type="m")
    asyncio.run(fam.main.defense_communication(ag, hook, ("do it", ["T"], "desc", "pad"), np.array(FULL3), 1))
    assert "Agent_2's reply: sanitised reply" in last_user(ag, 0)


@pytest.mark.parametrize("family", ["pi", "pi_gsm8k", "ma", "ta"])
def test_the_sender_keeps_its_own_memory_and_outbox_when_nothing_is_delivered(families, family):
    fam = families[family]
    fam.agents.allm_invoke = Recorder(reply_for(family))

    async def hook(turn, responses):
        return Plan({})

    ag = fam.agents.AgentGraphWithDefense(np.array(FULL3), ["s"] * 3, [], model_type="m")
    drive(fam, family, ag, hook, turns=2)
    for agent in ag.agents:
        assert [m["role"] for m in agent.memory] == ["system", "user", "assistant", "user", "assistant", "user",
                                                     "assistant"]
        assert len(agent.outbox) == 3                               # first answer + 2 updates
        assert agent.outbox[-1] == agent.last_response              # equal, but a copy (the sender's own record)


def test_outbox_entries_are_copies_a_sanitiser_cannot_alter_the_senders_record(families):
    fam = families["pi"]
    fam.agents.allm_invoke = Recorder()
    ag = graph(fam, FULL3)
    asyncio.run(ag.afirst_generate(("Q?", "A", "B")))
    sent = ag.agents[0].outbox[0]
    sent["reason"] = "tampered"
    assert ag.agents[0].last_response["reason"] != "tampered"


def test_external_context_reaches_only_the_agent_it_is_assigned_to(families):
    fam = families["pi"]
    rec = Recorder()
    fam.agents.allm_invoke = rec
    ag = graph(fam, FULL3)
    ag.context = {1: "\n\nDOCUMENT-FOR-1"}
    asyncio.run(ag.afirst_generate(("Q?", "A", "B")))
    firsts = [next(m for m in ag.agents[i].memory if m["role"] == "user")["content"] for i in range(3)]
    assert "DOCUMENT-FOR-1" in firsts[1] and "DOCUMENT-FOR-1" not in firsts[0] + firsts[2]


def test_system_prompts_are_exposed_by_the_family_graph_generator(families):
    for name, fam in families.items():
        data = fam.graph.generate_agent_graph_dataset(num_nodes=3, sparsity=1.0, num_graphs=1, num_attackers=1)[0]
        assert "{}" in data["normal_prompt"] and "{}" in data["attacker_prompt"], name


def test_ta_helpers_import_without_nltk_and_find_the_tools_file(families):
    fam = families["ta"]
    tools = fam.utils.get_tool_dict()
    assert len(tools) > 100
    out = fam.utils.evaluate_output_prompted("Thought: ok\nFinal Answer: nothing to do here at all today",
                                             target_tool="X", attacker_tools=["X"], user_tool="U",
                                             attacker_instruction="do X")
    assert out["eval"] in ("unsucc", "invalid")
