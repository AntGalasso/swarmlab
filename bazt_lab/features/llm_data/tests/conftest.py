"""Shared fixtures: tiny hand-written rows in the exact shapes of the pinned Hub datasets and vendored files."""
import hashlib
import json

import pytest

REV = {
    "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "csqa": "94630fe30dad47192a8546eb75f094926d47e155",
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
}


def mmlu_row(i):
    return {"question": f"mmlu question {i}", "subject": "elementary_mathematics",
            "choices": [f"m{i}a", f"m{i}b", f"m{i}c", f"m{i}d"], "answer": i % 4}


def csqa_row(i):
    return {"id": f"native{i}", "question": f"csqa question {i}", "question_concept": f"concept{i}",
            "choices": {"label": list("ABCDE"), "text": [f"c{i}{x}" for x in "abcde"]},
            "answerKey": "ABCDE"[i % 5]}


def gsm8k_row(i):
    return {"question": f"gsm8k question {i}", "answer": f"step one\nstep two <<1+1=2>>\n#### {i * 1000}"}


ROWS = {"mmlu": mmlu_row, "csqa": csqa_row, "gsm8k": gsm8k_row}


def injecagent_record(i):
    return {"Attacker Tools": [f"AttackerTool{i}"], "Modifed": 1, "Attacker Instruction": f"do bad thing {i}",
            "Expected Achievements": "harm", "Attack Type": "Physical Harm", "User Tool": f"UserTool{i}",
            "User Instruction": f"user instruction {i}", "Tool Parameters": "{'a': 1}",
            "Tool Response Template": "{'r': '<Attacker Instruction>'}", "Thought": f"Thought: {i}",
            "Tool Response": f"response {i}"}


def poisonedrag_record(i):
    return {"id": str(1000 + i), "question": f"rag question {i}", "correct answer": f"right{i}",
            "incorrect answer": f"wrong{i}", "adv_texts": [f"poison {i}.{j}" for j in range(5)]}


@pytest.fixture
def vendored_root(tmp_path):
    """A fake repo root holding the two vendored files at their G-Safeguard submodule paths."""
    ta = tmp_path / "vendor" / "G-safeguard" / "TA" / "datasets"
    ma = tmp_path / "vendor" / "G-safeguard" / "MA" / "datasets"
    ta.mkdir(parents=True)
    ma.mkdir(parents=True)
    (ta / "attack_unsucc_data.json").write_bytes(
        json.dumps([injecagent_record(i) for i in range(10)], indent=1).encode())
    (ma / "msmarco.json").write_bytes(
        json.dumps({str(1000 + i): poisonedrag_record(i) for i in range(10)}, indent=1).encode())
    return tmp_path


@pytest.fixture
def settings(vendored_root):
    base = vendored_root / "vendor" / "G-safeguard"
    sha = {"injecagent": hashlib.sha256((base / "TA/datasets/attack_unsucc_data.json").read_bytes()).hexdigest(),
           "poisonedrag_msmarco": hashlib.sha256((base / "MA/datasets/msmarco.json").read_bytes()).hexdigest()}
    return {"llm_data": {"revisions": dict(REV), "vendored_sha256": sha}}
