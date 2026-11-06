"""The child's last words: the summary line must not crash it when the parent is already gone."""
from __future__ import annotations

import json

from bazt_lab.features.sync.adapters import child


class ClosedPipe:
    def write(self, text):
        raise BrokenPipeError("parent is gone")

    def flush(self):
        raise BrokenPipeError("parent is gone")


class Capture:
    def __init__(self):
        self.text = ""

    def write(self, text):
        self.text += text

    def flush(self):
        pass


def test_the_summary_is_one_json_line():
    out = Capture()
    assert child.emit({"commits": 2, "failures": {"commit": 1}}, out) is True
    assert json.loads(out.text) == {"commits": 2, "failures": {"commit": 1}}


def test_a_closed_pipe_does_not_crash_the_child_after_its_final_push():
    assert child.emit({"commits": 2, "failures": {}}, ClosedPipe()) is False
