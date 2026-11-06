"""analysis.public: analyze() end to end on files written like the launcher's, and the command line."""
import json
import math
import random
import tomllib

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.analysis import public as analysis
from bazt_lab.features.analysis.tests import synth

U = "uncapped"
CELLS = [(0.1, 0.1), (0.2, 0.2)]


def dataset(gain=-0.15):
    """Mock frontier (A, B, C uncapped, C and BAZT capped) and an LLM capped block with no planted effect."""
    def asr(arm):
        return {"A": 0.6, "B": 0.5, "C": 0.4, "BAZT": 0.4 + gain}[arm]
    arms = {a: (lambda cell, topo, n, a=a: asr(a)) for a in ("A", "B", "C", "BAZT")}
    mock = synth.episodes(arms, cells_by_arm={"A": [(U, U)], "B": [(U, U)], "C": [(U, U)] + CELLS, "BAZT": CELLS},
                          topologies=("chain", "barabasi_albert"), sinks=range(2), noise_sd=0.01)
    llm = synth.episodes({"C": lambda *a: 0.4, "BAZT": lambda *a: 0.4}, block="llm_capped", platform="llm",
                         cells=[(0.1, 0.1)], topologies=("random_directed",), seeds=range(1, 6),
                         conversations=range(8), tasks=("mmlu",), backbones=("gpt",), noise_sd=0.01)
    return mock + llm


def settings_with_missing(*keys):
    tree = synth.settings()
    for key in keys:
        node = tree
        *path, leaf = key.split(".")
        for part in path:
            node = node[part]
        del node[leaf]
    return tree


def strict_json(text):
    def refuse(token):
        raise AssertionError(f"non-JSON token {token} in the report")
    return json.loads(text, parse_constant=refuse)


def test_analyze_returns_the_report_and_writes_it_as_strict_json(tmp_path):
    synth.write_rows(tmp_path, dataset())
    report = analysis.analyze(tmp_path, synth.settings())
    written = strict_json((tmp_path / "analysis_report.json").read_text())
    assert written["meta"]["profile"] == "test" and written["meta"]["pre_registered"] == {
        "ni_margin_accuracy": 0.01, "rank_agreement_threshold": 0.8}
    assert report["meta"]["platforms"]["mock"]["rows"] > 0 and report["meta"]["platforms"]["llm"]["rows"] > 0
    assert "mock_epidemiology" in report["meta"]["platforms"]["mock"]["blocks_missing"]
    budget = written["hypotheses"]["H-budget"]
    assert budget["mock"]["challengers"]["BAZT"]["summary"]["supported"] is True
    assert budget["llm"]["challengers"]["BAZT"]["summary"]["supported"] is False        # no planted effect there
    assert written["verdicts"]["H-budget"] == {"mock": {"BAZT": True}, "llm": {"BAZT": False}}
    assert written["verdicts"]["H-adaptive"]["mock"] == "no_data"


def test_every_analysis_of_the_spec_is_in_the_report(tmp_path):
    synth.write_rows(tmp_path, dataset())
    r = analysis.analyze(tmp_path, synth.settings())
    assert set(r["hypotheses"]) == {"H-budget", "H-efficiency", "H-topology", "H-adaptive"}
    assert set(r["secondary"]) == {"ladder", "ablations", "rank_agreement"}
    assert r["secondary"]["ladder"]["mock"]["holds"] is True
    assert r["secondary"]["ablations"]["mock"]["status"] == "no_data"
    assert r["secondary"]["rank_agreement"]["status"] == "not_evaluable"       # the LLM side lacks the A and B arms
    assert r["hypotheses"]["H-efficiency"]["mock"]["challengers"]["BAZT"]["frontiers"][0]["smallest"] is None \
        or r["hypotheses"]["H-efficiency"]["mock"]["challengers"]["BAZT"]["frontiers"][0]["smallest"]["audit"] == 0.1
    assert r["containment_per_cost"]["mock"]["entries"] and r["containment_per_cost"]["llm"]["status"] == "ok"


def test_the_report_is_the_same_whatever_the_order_of_the_rows_and_the_files(tmp_path):
    rows = dataset()
    shuffled = rows[:]
    random.Random(3).shuffle(shuffled)  # noqa: S311  test data
    synth.write_rows(tmp_path / "a", rows, per_file=40)
    synth.write_rows(tmp_path / "b", shuffled, per_file=17)
    first = analysis.analyze(tmp_path / "a", synth.settings())
    second = analysis.analyze(tmp_path / "b", synth.settings())
    first["meta"].pop("results_dir"), second["meta"].pop("results_dir")
    assert first == second


def test_a_missing_setting_refuses_with_its_name_before_anything_is_read(tmp_path):
    with pytest.raises(core.Unset) as e:
        analysis.analyze(tmp_path / "nowhere", settings_with_missing("runtime.ni_margin_accuracy"))
    assert e.value.args[0] == "runtime.ni_margin_accuracy"
    assert analysis.missing_keys(settings_with_missing("analysis.alpha", "profile")) == ["profile", "analysis.alpha"]


def test_a_results_directory_with_no_rows_gives_a_report_that_says_so(tmp_path):
    r = analysis.analyze(tmp_path, synth.settings(), out=tmp_path / "r.json")
    assert r["meta"]["platforms"]["mock"]["rows"] == 0
    assert r["hypotheses"]["H-budget"]["mock"]["status"] == "no_data"
    assert (tmp_path / "r.json").exists()


# ---------------------------------------------------------------------------------------- command line
def toml_text(tree):
    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        raise TypeError(v)

    def table(prefix, node, lines):
        scalars = {k: v for k, v in node.items() if not isinstance(v, dict)}
        if prefix and scalars:
            lines.append(f"[{prefix}]")
        lines += [f'"{k}" = {value(v)}' for k, v in scalars.items()]
        for k, v in node.items():
            if isinstance(v, dict):
                table(f"{prefix}.{k}" if prefix else k, v, lines)
    tree = {k: v for k, v in tree.items() if k != "profile"}
    lines = ['active_profile = "test"', "", "[profiles.test]"]
    table("", tree, lines)
    return "\n".join(lines)


def run_cli(tmp_path, tree, *args):
    settings = tmp_path / "profiles.toml"
    settings.write_text(toml_text(tree))
    return analysis.main(["--settings", str(settings), "--results-root", str(tmp_path / "root"), *args])


def test_the_command_line_analyses_the_profile_folder_and_prints_the_verdicts(tmp_path, capsys):
    synth.write_rows(tmp_path / "root" / "test", dataset())
    code = run_cli(tmp_path, synth.settings())
    out = capsys.readouterr().out
    assert code == 0 and "H-budget" in out and "mock" in out and "BAZT" in out
    assert strict_json((tmp_path / "root" / "test" / "analysis_report.json").read_text())["meta"]["profile"] == "test"


def test_the_command_line_accepts_overrides_and_an_output_path(tmp_path):
    synth.write_rows(tmp_path / "root" / "test", dataset())
    out = tmp_path / "elsewhere.json"
    code = run_cli(tmp_path, synth.settings(), "--out", str(out), "--set", 'analysis.blocks.budget.mock=["mock_nowhere"]')
    assert code == 0 and strict_json(out.read_text())["hypotheses"]["H-budget"]["mock"]["status"] == "no_data"


def test_the_command_line_names_every_missing_setting_and_writes_nothing(tmp_path, capsys):
    tree = synth.settings()
    del tree["runtime"]["ni_margin_accuracy"], tree["analysis"]["alpha"]
    code = run_cli(tmp_path, tree)
    err = capsys.readouterr().err
    assert code == 2 and "runtime.ni_margin_accuracy" in err and "analysis.alpha" in err
    assert not (tmp_path / "root").exists()


def test_a_report_value_that_is_not_a_number_is_null_not_nan(tmp_path):
    synth.write_rows(tmp_path, dataset())
    text = json.dumps(analysis.analyze(tmp_path, synth.settings()), allow_nan=False)      # would raise on NaN or inf
    assert "NaN" not in text and math.isfinite(len(text))


# ------------------------------------------------------------------------ strict JSON, integrity, refusals
def test_jsonable_turns_numpy_values_and_non_finite_numbers_into_plain_json():
    import numpy as np

    from bazt_lab.features.analysis.private import report
    out = report.jsonable({"a": np.float64(0.5), "n": np.int64(3), "flag": np.bool_(True), "bad": [math.nan, math.inf],
                           "t": (1, 2), "arr": np.array([1.0, math.nan]), 7: "key"})
    assert out == {"a": 0.5, "n": 3, "flag": True, "bad": [None, None], "t": [1, 2], "arr": [1.0, None], "7": "key"}
    assert type(out["n"]) is int and type(out["flag"]) is bool
    json.dumps(out, allow_nan=False)


def test_jsonable_refuses_a_value_it_cannot_represent_instead_of_stringifying_it():
    from bazt_lab.features.analysis.private import report
    with pytest.raises(TypeError):
        report.jsonable({"x": object()})


def test_the_report_carries_a_fingerprint_of_the_analysis_settings(tmp_path):
    synth.write_rows(tmp_path, dataset())
    first = analysis.analyze(tmp_path, synth.settings(), out=tmp_path / "1.json")["meta"]["settings_sha256"]
    again = analysis.analyze(tmp_path, synth.settings(), out=tmp_path / "2.json")["meta"]["settings_sha256"]
    other = analysis.analyze(tmp_path, synth.settings(analysis__efficiency_tolerance=0.01),
                             out=tmp_path / "3.json")["meta"]["settings_sha256"]
    margin = analysis.analyze(tmp_path, synth.settings(runtime__ni_margin_accuracy=0.02),
                              out=tmp_path / "4.json")["meta"]["settings_sha256"]
    assert first == again and len(first) == 64
    assert other != first and margin != first


def write_manifest(root, block, **pre_registered):
    (root / block).mkdir(parents=True, exist_ok=True)
    (root / block / "manifest-2026-11-06T10-00-00.json").write_text(json.dumps({"block": block, "pre_registered": pre_registered}))


def test_manifests_that_recorded_the_same_pre_registered_values_are_consistent(tmp_path):
    synth.write_rows(tmp_path, dataset())
    write_manifest(tmp_path, "mock_frontier", ni_margin_accuracy=0.01, rank_agreement_threshold=0.8)
    m = analysis.analyze(tmp_path, synth.settings())["meta"]["manifests"]
    assert m == {"checked": 1, "pre_registered_mismatch": []}


def test_a_pre_registered_value_changed_after_the_run_is_reported_with_the_block_and_both_values(tmp_path):
    synth.write_rows(tmp_path, dataset())
    write_manifest(tmp_path, "mock_frontier", ni_margin_accuracy=0.02, rank_agreement_threshold=0.8)
    m = analysis.analyze(tmp_path, synth.settings())["meta"]["manifests"]
    assert m["checked"] == 1 and m["pre_registered_mismatch"] == [
        {"block": "mock_frontier", "file": "manifest-2026-11-06T10-00-00.json", "key": "ni_margin_accuracy",
         "manifest": 0.02, "settings": 0.01}]


def test_an_unreadable_manifest_is_reported_not_skipped(tmp_path):
    synth.write_rows(tmp_path, dataset())
    (tmp_path / "mock_frontier" / "manifest-broken.json").write_text("{not json")
    m = analysis.analyze(tmp_path, synth.settings())["meta"]["manifests"]
    assert m["pre_registered_mismatch"] == [{"block": "mock_frontier", "file": "manifest-broken.json",
                                             "key": "manifest", "manifest": "unreadable", "settings": None}]


def test_a_results_directory_that_does_not_exist_is_refused_and_nothing_is_written(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere"):
        analysis.analyze(tmp_path / "nowhere", synth.settings())
    assert not (tmp_path / "nowhere").exists()


def test_the_report_is_written_atomically_and_replaces_the_previous_one(tmp_path):
    synth.write_rows(tmp_path, dataset())
    out = tmp_path / "report.json"
    out.write_text("stale")
    analysis.analyze(tmp_path, synth.settings(), out=out)
    assert strict_json(out.read_text())["meta"]["profile"] == "test"
    assert not list(tmp_path.glob("*.tmp"))


def test_the_command_line_refuses_a_missing_results_directory_with_its_path(tmp_path, capsys):
    code = run_cli(tmp_path, synth.settings())
    assert code == 2 and str(tmp_path / "root" / "test") in capsys.readouterr().err


def test_the_command_line_warns_when_a_pre_registered_value_differs_from_its_manifest(tmp_path, capsys):
    synth.write_rows(tmp_path / "root" / "test", dataset())
    write_manifest(tmp_path / "root" / "test", "mock_frontier", ni_margin_accuracy=0.02, rank_agreement_threshold=0.8)
    assert run_cli(tmp_path, synth.settings()) == 0
    assert "ni_margin_accuracy" in capsys.readouterr().out


def test_the_settings_block_of_the_contract_names_every_analysis_key_the_code_requires():
    from pathlib import Path

    from bazt_lab.features.analysis.private import config
    text = (Path(analysis.__file__).parent / "CONTRACT.md").read_text(encoding="utf-8")
    block = text.split("```toml", 1)[1].split("```", 1)[0]
    assert tomllib.loads(block)["analysis"]["pair_unit"] == {"mock": "seed", "llm": "episode"}     # valid TOML
    for key in config.REQUIRED_KEYS:
        if key.startswith("analysis."):
            assert key.split(".")[-1] in block, f"{key} is missing from the settings block of CONTRACT.md"


def test_python_dash_m_runs_the_public_command_line():
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-m", "bazt_lab.features.analysis", "--help"], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 0 and "--results-root" in done.stdout
