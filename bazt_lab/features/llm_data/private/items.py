"""Typed task items, the row builders for the Hub tasks, and the answer checkers (spec 4.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional

TASKS = ("mmlu", "csqa", "gsm8k", "injecagent", "poisonedrag_msmarco")

# Dataset id, config and split are part of the task definition (spec 4.1); only the revision is a pinned setting.
HUB = {"mmlu": ("cais/mmlu", "all", "test"),
       "csqa": ("tau/commonsense_qa", "default", "validation"),      # the public test split has no labels
       "gsm8k": ("openai/gsm8k", "main", "test")}

LETTERS = "ABCDE"


class NoAnswerKey(ValueError):
    """The task has no ground-truth answer to check against."""


@dataclass(frozen=True)
class Source:
    """Where an item came from. Hub: dataset, config, split and the commit sha `revision`. Vendored file: the path
    in `dataset`, `split` = "all" / "train" / "test", and `sha256` of the file (CRLF read as LF)."""
    dataset: str
    config: Optional[str]
    split: Optional[str]
    revision: Optional[str]
    sha256: Optional[str]


@dataclass(frozen=True)
class TaskItem:
    task: str
    id: str                          # "<task>:<position in the pinned split or file>", stable under the pin
    question: str
    choices: tuple                   # () when the task has none
    answer: str                      # MMLU / CSQA: the letter; GSM8K: the number after "####"; PoisonedRAG: the
                                     # correct answer text; "" for InjecAgent (no answer key)
    source: Source
    meta: dict = field(default_factory=dict)


def _mmlu(row: dict, source: Source) -> TaskItem:
    choices, answer = tuple(row["choices"]), row["answer"]
    if len(choices) != 4 or type(answer) is not int or not 0 <= answer < 4:
        raise ValueError(f"mmlu row {row['_row']}: need 4 choices and an answer index 0-3, got "
                         f"{len(choices)} choices and {answer!r}")
    return TaskItem("mmlu", f"mmlu:{row['_row']}", row["question"], choices, LETTERS[answer], source,
                    {"subject": row["subject"], "answer_index": answer})


def _csqa(row: dict, source: Source) -> TaskItem:
    labels, texts, key = list(row["choices"]["label"]), tuple(row["choices"]["text"]), row["answerKey"]
    if labels != list(LETTERS[:len(labels)]) or len(texts) != len(labels) or key not in labels:
        raise ValueError(f"csqa row {row['_row']}: labels {labels}, {len(texts)} texts, answerKey {key!r}")
    return TaskItem("csqa", f"csqa:{row['_row']}", row["question"], texts, key, source,
                    {"native_id": row["id"], "question_concept": row["question_concept"]})


def _gsm8k(row: dict, source: Source) -> TaskItem:
    solution = row["answer"]
    number = solution.split("####")[-1].strip().replace(",", "") if "####" in solution else ""
    if _number(number) is None:
        raise ValueError(f"gsm8k row {row['_row']}: no number after '####' in {solution[-40:]!r}")
    return TaskItem("gsm8k", f"gsm8k:{row['_row']}", row["question"], (), number, source, {"solution": solution})


_BUILD = {"mmlu": _mmlu, "csqa": _csqa, "gsm8k": _gsm8k}


def hub_item(task: str, row: dict, revision: str) -> TaskItem:
    """An item from one row of `adapters.hf.draw`."""
    dataset, config, split = HUB[task]
    try:
        return _BUILD[task](row, Source(dataset, config, split, revision, None))
    except KeyError as missing:
        raise ValueError(f"{task} row {row.get('_row')}: field {missing} is missing") from None
    except TypeError as err:                    # a field of another type than the pinned card declares
        raise ValueError(f"{task} row {row.get('_row')}: unexpected field type ({err})") from None


def _number(value) -> Optional[Decimal]:
    """A finite decimal from an int, float or text such as "$1,234." ; None when it is not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip().replace(",", "").lstrip("$").rstrip(".").strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _letter(answer, n: int, index_ok: bool) -> Optional[str]:
    letters = LETTERS[:n]
    if isinstance(answer, int) and not isinstance(answer, bool):
        return letters[answer] if index_ok and 0 <= answer < n else None
    if isinstance(answer, str):
        text = answer.strip().strip("().:").strip().upper()
        if text in tuple(letters):
            return text
        if index_ok and text in tuple(map(str, range(n))):
            return letters[int(text)]
    return None


def check_answer(item: TaskItem, answer) -> bool:
    """Is `answer` the item's correct answer? MMLU: letter or index; CSQA: letter; GSM8K: the number (commas, a
    leading "$" and a trailing "." are ignored; 72 == "72.0"). Tasks without an answer key raise NoAnswerKey."""
    if item.task in ("mmlu", "csqa"):
        return _letter(answer, len(item.choices), index_ok=item.task == "mmlu") == item.answer
    if item.task == "gsm8k":
        got = _number(answer)
        return got is not None and got == Decimal(item.answer)
    raise NoAnswerKey(f"{item.task} has no answer key; its success criterion is not a ground-truth answer")


def render_question(item: TaskItem) -> str:
    """The question with its choices as "A. text" lines, the layout of G-Safeguard's `format_example`."""
    return item.question + "".join(f"\n{LETTERS[i]}. {text}" for i, text in enumerate(item.choices))
