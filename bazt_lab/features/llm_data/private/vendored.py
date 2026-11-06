"""InjecAgent (TA) and PoisonedRAG over MS MARCO (MA): the files vendored in the G-Safeguard submodule.

Neither has a Hub dataset. The submodule is pinned (D33); its files are further pinned by sha256 from settings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

from bazt_lab.features.llm_data.private.items import Source, TaskItem

REPO = Path(__file__).resolve().parents[4]
FILES = {"injecagent": "vendor/G-safeguard/TA/datasets/attack_unsucc_data.json",
         "poisonedrag_msmarco": "vendor/G-safeguard/MA/datasets/msmarco.json"}


class DatasetMismatch(ValueError):
    """A vendored file's sha256 differs from the one pinned in settings."""


def file_sha256(data: bytes) -> str:
    """sha256 over the content with CRLF read as LF: git on Windows (core.autocrlf) rewrites text files."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def read(task: str, expected: str, root: Optional[Path] = None) -> Tuple[List[Tuple[str, dict]], str]:
    """The file's records as (position or native id, record) in file order, and its verified sha256."""
    path = Path(root or REPO) / FILES[task]
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} is missing; run `git submodule update --init vendor/G-safeguard`") from None
    digest = file_sha256(data)
    if digest != expected:
        raise DatasetMismatch(f"{FILES[task]}: sha256 {digest} differs from {expected} pinned in "
                              f"llm_data.vendored_sha256.{task}")
    content = json.loads(data)
    if task == "injecagent":
        if not isinstance(content, list) or not all(isinstance(r, dict) and "User Instruction" in r for r in content):
            raise ValueError("injecagent: expected a list of records with a 'User Instruction'")
        return [(str(i), record) for i, record in enumerate(content)], digest
    if not isinstance(content, dict) or not all(isinstance(r, dict) and {"question", "correct answer",
                                                "incorrect answer", "adv_texts"} <= r.keys() for r in content.values()):
        raise ValueError("poisonedrag_msmarco: expected an object of records with question, correct answer, "
                         "incorrect answer and adv_texts")
    return [(str(key), record) for key, record in content.items()], digest


def source(task: str, part: Optional[str], digest: str) -> Source:
    return Source(FILES[task], None, part or "all", None, digest)


def item(task: str, native_id: str, record: dict, src: Source) -> TaskItem:
    if task == "injecagent":
        return TaskItem(task, f"{task}:{native_id}", record["User Instruction"], (), "", src, {"record": record})
    return TaskItem(task, f"{task}:{native_id}", record["question"], (), record["correct answer"], src,
                    {"incorrect_answer": record["incorrect answer"], "adv_texts": record["adv_texts"],
                     "native_id": native_id})
