"""Failures during a run are reported on stderr, once per operation, never silently and never fatally."""
from __future__ import annotations

from bazt_lab.features.tracking.private.failures import FailureLog


def test_the_first_failure_of_an_operation_is_printed_to_stderr(capsys):
    log = FailureLog()
    log.report("log", ConnectionError("network down"))
    out, err = capsys.readouterr()
    assert out == ""
    assert "log" in err and "ConnectionError" in err and "network down" in err and "disk" in err


def test_repeats_of_the_same_operation_are_counted_but_not_printed_again(capsys):
    log = FailureLog()
    for _ in range(5):
        log.report("log", ConnectionError("network down"))
    assert capsys.readouterr().err.count("network down") == 1
    assert log.counts == {"log": 5}


def test_another_operation_is_reported_in_its_own_right(capsys):
    log = FailureLog()
    log.report("log", ValueError("a"))
    log.report("artifact", OSError("b"))
    err = capsys.readouterr().err
    assert "a" in err and "b" in err
    assert log.counts == {"log": 1, "artifact": 1}


def test_the_redactor_is_applied_to_everything_printed(capsys):
    log = FailureLog(redact=lambda text: text.replace("hunter2", "[redacted]"))
    log.report("log", RuntimeError("bad token hunter2"))
    err = capsys.readouterr().err
    assert "hunter2" not in err and "[redacted]" in err


def test_the_summary_is_none_without_failures_and_counts_them_otherwise():
    log = FailureLog()
    assert log.summary() is None
    log.report("log", ValueError("x"))
    log.report("log", ValueError("x"))
    log.report("log_rows[t]", ValueError("y"))
    text = log.summary()
    assert "3" in text and "log: 2" in text and "log_rows[t]: 1" in text and "incomplete" in text
