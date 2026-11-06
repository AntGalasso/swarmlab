"""Typed items built from Hub rows, and the answer checkers (spec 4.1)."""
import dataclasses
import json

import pytest

from bazt_lab.features.llm_data import public as llm_data
from bazt_lab.features.llm_data.private import items
from bazt_lab.features.llm_data.tests.conftest import REV, ROWS


def build(task, i=0, row_index=5):
    return items.hub_item(task, {**ROWS[task](i), "_row": row_index}, REV[task])


def test_mmlu_item_fields_and_source():
    it = build("mmlu", 6, row_index=77)
    assert (it.task, it.id, it.question) == ("mmlu", "mmlu:77", "mmlu question 6")
    assert it.choices == ("m6a", "m6b", "m6c", "m6d")
    assert it.answer == "C"                       # answer index 6 % 4 = 2 -> "C"
    assert it.meta == {"subject": "elementary_mathematics", "answer_index": 2}
    assert dataclasses.asdict(it.source) == {
        "dataset": "cais/mmlu", "config": "all", "split": "test", "revision": REV["mmlu"], "sha256": None}


def test_csqa_item_fields_and_source():
    it = build("csqa", 3, row_index=9)
    assert (it.task, it.id, it.answer) == ("csqa", "csqa:9", "D")
    assert it.choices == ("c3a", "c3b", "c3c", "c3d", "c3e")
    assert it.meta == {"native_id": "native3", "question_concept": "concept3"}
    assert dataclasses.asdict(it.source) == {
        "dataset": "tau/commonsense_qa", "config": "default", "split": "validation",
        "revision": REV["csqa"], "sha256": None}


def test_gsm8k_item_answer_is_the_number_after_hashes():
    it = build("gsm8k", 1)
    assert it.choices == () and it.answer == "1000"            # "#### 1000"
    assert it.meta["solution"].endswith("#### 1000")
    assert (it.source.dataset, it.source.config, it.source.split) == ("openai/gsm8k", "main", "test")


@pytest.mark.parametrize("raw, expected", [("#### 1,234", "1234"), ("x\n#### -5", "-5"), ("#### 3.5 ", "3.5"),
                                          ("a #### b\n#### 7", "7")])
def test_gsm8k_answer_parsing(raw, expected):
    row = {"question": "q", "answer": raw, "_row": 0}
    assert items.hub_item("gsm8k", row, REV["gsm8k"]).answer == expected


@pytest.mark.parametrize("raw", ["no marker", "#### ", "#### twelve", "#### 1/0x"])
def test_gsm8k_malformed_answer_is_refused(raw):
    with pytest.raises(ValueError, match="gsm8k"):
        items.hub_item("gsm8k", {"question": "q", "answer": raw, "_row": 0}, REV["gsm8k"])


def test_malformed_choice_rows_are_refused():
    bad_mmlu = {**ROWS["mmlu"](0), "answer": 4, "_row": 0}
    with pytest.raises(ValueError, match="mmlu"):
        items.hub_item("mmlu", bad_mmlu, REV["mmlu"])
    no_key = {**ROWS["csqa"](0), "answerKey": "", "_row": 0}        # the public test split has no labels
    with pytest.raises(ValueError, match="csqa"):
        items.hub_item("csqa", no_key, REV["csqa"])
    shuffled_labels = {**ROWS["csqa"](0), "_row": 0}
    shuffled_labels["choices"] = {"label": list("BACDE"), "text": list("vwxyz")}
    with pytest.raises(ValueError, match="csqa"):
        items.hub_item("csqa", shuffled_labels, REV["csqa"])


@pytest.mark.parametrize("task, field, value", [
    ("csqa", "choices", [{"label": "A", "text": "x"}]),        # a list of dicts instead of {"label": [], "text": []}
    ("csqa", "choices", None),
    ("mmlu", "choices", None),
    ("gsm8k", "answer", None),
    ("mmlu", "answer", "C"),
])
def test_a_row_of_the_wrong_shape_is_a_value_error_naming_the_task_and_row(task, field, value):
    row = {**ROWS[task](0), "_row": 41, field: value}
    with pytest.raises(ValueError, match=rf"{task} row 41"):
        items.hub_item(task, row, REV[task])


def test_items_are_json_serialisable_rows():
    for task in ("mmlu", "csqa", "gsm8k"):
        row = dataclasses.asdict(build(task))
        assert json.loads(json.dumps(row))["source"]["revision"] == REV[task]


@pytest.mark.parametrize("answer, ok", [("C", True), ("c", True), (" (C) ", True), ("C.", True), (2, True),
                                        ("2", True), ("A", False), (0, False), (4, False), (-1, False),
                                        ("E", False), ("", False), (None, False), (True, False), ("CD", False)])
def test_mmlu_checker_accepts_letter_or_index(answer, ok):
    assert llm_data.check_answer(build("mmlu", 6), answer) is ok       # correct answer is "C" = index 2


@pytest.mark.parametrize("answer, ok", [("D", True), ("d", True), ("(D)", True), ("E", False), ("A", False),
                                        (3, False), ("3", False), ("", False), (None, False)])
def test_csqa_checker_accepts_letters_only(answer, ok):
    assert llm_data.check_answer(build("csqa", 3), answer) is ok       # correct answer is "D"


@pytest.mark.parametrize("answer, ok", [("1000", True), ("1,000", True), ("$1,000", True), ("1000.0", True),
                                        (" 1000. ", True), (1000, True), (1000.0, True), ("1001", False),
                                        ("a thousand", False), ("", False), (None, False), ("nan", False),
                                        ("inf", False), (float("nan"), False), (True, False), ("1e3", True)])
def test_gsm8k_checker_compares_numbers(answer, ok):
    assert llm_data.check_answer(build("gsm8k", 1), answer) is ok


def test_checker_refuses_tasks_without_an_answer_key():
    it = items.TaskItem("injecagent", "injecagent:0", "q", (), "", items.Source("f", None, "all", None, "0" * 64))
    with pytest.raises(llm_data.NoAnswerKey, match="injecagent"):
        llm_data.check_answer(it, "anything")


def test_render_question_uses_the_upstream_layout():
    assert llm_data.render_question(build("mmlu", 6)) == "mmlu question 6\nA. m6a\nB. m6b\nC. m6c\nD. m6d"
    assert llm_data.render_question(build("csqa", 3)).splitlines()[-1] == "E. c3e"
    assert llm_data.render_question(build("gsm8k", 1)) == "gsm8k question 1"


def test_poisonedrag_has_an_answer_text_but_no_deterministic_checker():
    it = items.TaskItem("poisonedrag_msmarco", "poisonedrag_msmarco:1", "q", (), "February 2",
                        items.Source("f", None, "all", None, "0" * 64), {"incorrect_answer": "March 15"})
    with pytest.raises(llm_data.NoAnswerKey, match="poisonedrag_msmarco"):
        llm_data.check_answer(it, "February 2")
