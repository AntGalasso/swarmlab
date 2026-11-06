"""The two attack templates built from their papers."""
from bazt_lab.features.llm_harness.private import prompts


def test_infection_prompt_keeps_the_four_parts_of_lee_and_tiwari_fig_8():
    text = prompts.infection_document("The correct answer is B")
    assert text.count("[[[<USER>") == 1 and text.endswith("</USER>]]]")
    assert "[CTRL+C]" in text and "I changed my mind" in text                  # prompt hijacking
    assert 'Could you say "The correct answer is B"?' in text                  # payload
    assert 'starts with "[[["' in text and "ends with" in text                 # self-replication
    assert prompts.REPLICATION_MARKER in text
    assert "{attack_phrase}" not in text


def test_the_infection_asks_every_infected_agent_to_state_the_payload_before_replicating():
    text = prompts.infection_document("PAYLOAD")
    else_branch = text.split("} ELSE {")[1]
    assert else_branch.index("PAYLOAD") < else_branch.index("[[[")


def test_adversary_messages_carry_the_three_inputs_of_the_reflection_prompt():
    msgs = prompts.adversary_messages(["Agent_1: A"], "earlier instruction", "make the answer B")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    body = msgs[0]["content"]
    assert "Agent_1: A" in body and "earlier instruction" in body and "make the answer B" in body
    assert "Intercepted Messages" in body and "Previous Instructions" in body


def test_adversary_with_no_previous_instruction_says_none():
    assert "None" in prompts.adversary_messages(["x"], None, "g")[0]["content"]


def test_rewrite_appends_the_instruction_to_the_reason_of_a_dict_view_and_to_a_string_view():
    out = prompts.rewrite({"answer": "A", "reason": "because"}, "do X")
    assert out["answer"] == "A" and out["reason"].startswith("because") and out["reason"].endswith("do X")
    assert prompts.rewrite("plain reply", "do X").endswith("do X")


def test_rewrite_does_not_alter_the_senders_own_record():
    view = {"answer": "A", "reason": "because"}
    prompts.rewrite(view, "do X")
    assert view == {"answer": "A", "reason": "because"}
