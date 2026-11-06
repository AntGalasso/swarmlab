"""The import gate (bazt_lab/tools/check_architecture.py) on planted violations and on the real tree.

Every rule of docs/ARCHITECTURE_EXPERIMENTS.md has a clean case (must pass) and a planted case (must fail with exactly
that rule). Planted trees are dicts {path: text} given to `check_sources`, so no case touches the disk; a few tests
write real trees under tmp_path to cover the loader and the command line. The real tree is checked last.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bazt_lab.tools import check_architecture as gate

REPO = Path(__file__).resolve().parents[1]

FEATURE_FILES = {
    "__init__.py": "",
    "public.py": "ARMS = ()\n",
    "CONTRACT.md": "# contract\n",
    "private/__init__.py": "",
    "adapters/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_basic.py": "def test_basic():\n    assert True\n",
}
WIRING = 'import importlib\n\n\ndef feature(name):\n    return importlib.import_module(f"bazt_lab.features.{name}.public")\n'


def tree(features=("a", "b"), extra: dict | None = None, omit: tuple = ()) -> dict:
    """A clean tree: bazt_lab/{__init__, app, core} and well-formed features; `extra` overrides or adds files."""
    files = {
        "bazt_lab/__init__.py": "",
        "bazt_lab/app.py": WIRING,
        "bazt_lab/core/__init__.py": "",
        "bazt_lab/core/public.py": "from bazt_lab.core.private import rng\n",
        "bazt_lab/core/private/__init__.py": "",
        "bazt_lab/core/private/rng.py": "import torch\n",
        "bazt_lab/features/__init__.py": "",
    }
    for name in features:
        for rel, text in FEATURE_FILES.items():
            if f"{name}/{rel}" not in omit:
                files[f"bazt_lab/features/{name}/{rel}"] = text
    files.update(extra or {})
    return files


def rules(files: dict) -> set[str]:
    return {v.rule for v in gate.check_sources(files)}


def planted(rel: str, source: str, features=("a", "b")) -> set[str]:
    return rules(tree(features, {rel: source}))


def write(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


# ------------------------------------------------------------------------------------------- clean trees
def test_clean_synthetic_tree_has_no_violation():
    files = tree(("a", "b", "c", "d"), {
        "bazt_lab/features/a/private/impl.py": "import torch\nimport numpy as np\nfrom bazt_lab.core import public as core\n",
        "bazt_lab/features/a/adapters/hub.py": "import wandb\nfrom huggingface_hub import HfApi\nfrom dotenv import load_dotenv\n",
        "bazt_lab/tools/__init__.py": "",
    })
    assert gate.check_sources(files) == []


@pytest.mark.parametrize("line", [
    "from bazt_lab.features.b import public as b",
    "from bazt_lab.features.b.public import ARMS",
    "import bazt_lab.features.b.public as b",
    "from ...b import public",
])
def test_cross_feature_import_through_public_is_allowed(line):
    assert planted("bazt_lab/features/a/private/x.py", line + "\n") == set()


@pytest.mark.parametrize("line", [
    "from bazt_lab.features.b.private import impl",
    "from bazt_lab.features.b.private.impl import thing",
    "import bazt_lab.features.b.private.impl",
    "from bazt_lab.features.b.adapters import hub",
    "from bazt_lab.features.b import thing",
    "from bazt_lab.features.b import public, private",
    "from bazt_lab.features import b",
    "from bazt_lab.features import *",
    "import bazt_lab.features.b",
    "from ...b.private import impl",
    "from ...b import private",
])
def test_cross_feature_import_beyond_public_is_refused(line):
    assert planted("bazt_lab/features/a/private/x.py", line + "\n") == {"feature-public"}


def test_a_feature_is_free_inside_itself():
    files = tree(extra={
        "bazt_lab/features/a/public.py": "from .private import impl\nfrom bazt_lab.features.a.adapters import hub\n",
        "bazt_lab/features/a/tests/test_basic.py": "from bazt_lab.features.a.private import impl\n",
        "bazt_lab/features/a/private/impl.py": "from ..adapters import hub\n",
    })
    assert gate.check_sources(files) == []


# ------------------------------------------------------------------------------------------------- core
def test_core_imports_no_feature():
    assert planted("bazt_lab/core/private/x.py", "from bazt_lab.features.a import public\n") == {"core-isolated"}


@pytest.mark.parametrize("line", ["import bazt_lab.app", "from bazt_lab import app", "from bazt_lab.app import main"])
@pytest.mark.parametrize("where", ["bazt_lab/core/private/x.py", "bazt_lab/features/a/private/x.py",
                                   "bazt_lab/features/a/tests/t.py", "bazt_lab/tools/x.py"])
def test_nothing_but_app_itself_imports_bazt_lab_app(where, line):
    assert planted(where, line + "\n") == {"app-root"}


@pytest.mark.parametrize("line", [
    "from bazt_lab.core import public as core",
    "from bazt_lab.core.public import get",
    "import bazt_lab.core.public as core",
    "from .....core import public",
])
def test_core_is_reached_through_public(line):
    assert planted("bazt_lab/features/a/private/deeper/x.py", line + "\n") == set()


@pytest.mark.parametrize("line", [
    "from bazt_lab.core.private.rng import stream",
    "from bazt_lab.core import private",
    "from bazt_lab.core import checkpoint",
    "import bazt_lab.core.private.settings",
    "from bazt_lab.core import *",
    "import bazt_lab.core",
    "from bazt_lab import core",
    "from .....core.private import rng",
])
def test_core_private_is_refused_outside_core(line):
    assert planted("bazt_lab/features/a/private/deeper/x.py", line + "\n") == {"core-public"}


def test_core_is_free_inside_itself():
    files = tree(extra={"bazt_lab/core/tests/__init__.py": "",
                        "bazt_lab/core/tests/test_x.py": "from bazt_lab.core.private import rng\n"})
    assert gate.check_sources(files) == []


# ------------------------------------------------------------------------------------------ only app.py
def test_only_app_imports_every_feature():
    names = ("a", "b", "c", "d")
    every = "".join(f"from bazt_lab.features.{n} import public as {n}\n" for n in "bcd")
    assert planted("bazt_lab/features/a/private/wiring.py", every, names) == {"god-feature"}
    assert planted("bazt_lab/features/a/private/wiring.py", every.splitlines(True)[0], names) == set()
    assert planted("bazt_lab/tools/wiring.py", "".join(
        f"from bazt_lab.features.{n} import public as {n}\n" for n in names), names) == {"god-feature"}
    app = "".join(f"from bazt_lab.features.{n} import public as {n}\n" for n in names)
    assert planted("bazt_lab/app.py", app, names) == set()


# ------------------------------------------------------------------------------------ third-party services
SERVICES = ["wandb", "datasets", "huggingface_hub", "openai", "sentence_transformers", "torch_geometric", "swarmlab",
            "bazt", "llm_client"]
NOT_ADAPTERS = ["bazt_lab/features/a/private/x.py", "bazt_lab/features/a/public.py",
                "bazt_lab/features/a/tests/test_x.py", "bazt_lab/core/private/x.py", "bazt_lab/app.py",
                "bazt_lab/tools/x.py"]


@pytest.mark.parametrize("form", ["import {s}\n", "from {s}.sub import thing\n", "import {s}.sub as s\n"])
@pytest.mark.parametrize("service", SERVICES)
def test_services_are_importable_only_in_adapters(service, form):
    source = form.format(s=service)
    assert planted("bazt_lab/features/a/adapters/x.py", source) == set()
    for where in NOT_ADAPTERS:
        assert planted(where, source) == {"service-adapter"}, where


def test_torch_numpy_and_the_package_itself_are_not_services():
    source = "import torch\nimport numpy as np\nimport bazt_lab.core.public\nfrom bazt_lab.core import public\n"
    assert planted("bazt_lab/features/a/private/x.py", source) == set()


@pytest.mark.parametrize("source, expected", [
    ('import importlib\nimportlib.import_module("wandb")\n', {"service-adapter"}),
    ('from importlib import import_module\nimport_module("bazt.runner")\n', {"service-adapter"}),
    ('from importlib import import_module as load\nload("openai")\n', {"service-adapter"}),
    ('import importlib as il\nil.import_module("wandb")\n', {"service-adapter"}),
    ('__import__("openai")\n', {"service-adapter"}),
    ('import importlib\nimportlib.import_module("bazt_lab.features.b.private.impl")\n', {"feature-public"}),
    ('import importlib\nimportlib.import_module("bazt_lab.features.b.public")\n', set()),
    ('import importlib\nimportlib.import_module(name)\n', {"dynamic-import"}),
    ('from importlib import import_module as load\nload(name)\n', {"dynamic-import"}),
    ('import importlib\nimportlib.import_module(".sub", __package__)\n', {"dynamic-import"}),
    ('import importlib\nimportlib.import_module(f"bazt_lab.features.{n}.public")\n', {"dynamic-import"}),
    ('import importlib\nimportlib.import_module(name)  # arch: allow dynamic-import\n', {"dynamic-import"}),
])
def test_dynamic_imports_are_checked(source, expected):
    assert planted("bazt_lab/features/a/private/x.py", source) == expected


def test_adapters_may_load_modules_dynamically():
    source = "import importlib\nimportlib.import_module(name)\nimportlib.import_module('wandb')\n"
    assert planted("bazt_lab/features/a/adapters/x.py", source) == set()


def test_the_composition_root_wires_features_by_their_public_module_only():
    wire = 'import importlib\nimportlib.import_module(f"bazt_lab.features.{n}.public")\n'
    assert planted("bazt_lab/app.py", wire) == set()
    for bad in ('f"bazt_lab.features.{n}.private"', 'f"bazt_lab.features.{n}"', 'f"bazt_lab.{n}.public"', "name",
                'f"bazt_lab.features.{n}.public.{m}"'):
        assert planted("bazt_lab/app.py", f"import importlib\nimportlib.import_module({bad})\n") == {"dynamic-import"}, bad
    assert planted("bazt_lab/features/a/private/x.py", wire) == {"dynamic-import"}


# -------------------------------------------------------------------------------------------- feature shape
@pytest.mark.parametrize("missing", list(FEATURE_FILES))
def test_every_part_of_the_feature_shape_is_required(missing):
    found = gate.check_sources(tree(omit=(f"a/{missing}",)))
    assert {v.rule for v in found} == {"shape"}, found
    assert all("features/a" in v.path for v in found)


@pytest.mark.parametrize("missing", ["bazt_lab/__init__.py", "bazt_lab/app.py", "bazt_lab/core/__init__.py",
                                     "bazt_lab/core/public.py", "bazt_lab/core/private/__init__.py"])
def test_the_package_skeleton_is_required(missing):
    files = tree()
    del files[missing]
    found = gate.check_sources(files)
    assert {v.rule for v in found} == {"shape"} and any(missing in v.message or missing in v.path for v in found), found


def test_contract_must_not_be_empty():
    assert planted("bazt_lab/features/a/CONTRACT.md", "  \n") == {"shape"}


def test_tests_directory_needs_a_test_module():
    files = tree(omit=("a/tests/test_basic.py",), extra={"bazt_lab/features/a/tests/helpers.py": ""})
    assert rules(files) == {"shape"}
    nested = tree(omit=("a/tests/test_basic.py",), extra={"bazt_lab/features/a/tests/sub/test_deep.py": ""})
    assert rules(nested) == set()


def test_code_outside_public_private_adapters_tests_is_refused():
    assert planted("bazt_lab/features/a/helpers.py", "x = 1\n") == {"shape"}
    assert planted("bazt_lab/features/stray.py", "x = 1\n") == {"shape"}
    assert planted("bazt_lab/utils.py", "x = 1\n") == {"shape"}


def test_data_files_inside_a_feature_are_fine():
    assert planted("bazt_lab/features/a/data/items.json", "[]") == set()
    assert planted("bazt_lab/features/a/private/data/items.json", "[]") == set()


def test_a_feature_name_that_starts_with_an_underscore_or_dot_is_not_a_feature():
    assert planted("bazt_lab/features/_template/notes.txt", "x") == set()


# ------------------------------------------------------------------------------------------ feature __main__.py
THIN_MAIN = "import sys\n\nfrom bazt_lab.features.a.public import main\n\nsys.exit(main())\n"


@pytest.mark.parametrize("source", [
    THIN_MAIN,
    "import sys\n\nfrom bazt_lab.features.a import public\n\nif __name__ == '__main__':\n    sys.exit(public.main())\n",
    "from bazt_lab.features.a.public import main\n\nmain()\n",
])
def test_a_thin_feature_main_is_allowed(source):
    assert planted("bazt_lab/features/a/__main__.py", source) == set()


@pytest.mark.parametrize("source, expected", [
    ("import sys\nfrom bazt_lab.features.a.private import impl\nsys.exit(impl.main())\n", {"shape"}),
    ("import sys\nfrom bazt_lab.features.a.public import main\nx = 1\nsys.exit(main())\n", {"shape"}),
    ("import sys\nfrom bazt_lab.features.a.public import main\n\ndef run():\n    sys.exit(main())\n", {"shape"}),
    ("import wandb\nfrom bazt_lab.features.a.public import main\nmain()\n", {"service-adapter", "shape"}),
    ("import os\nfrom bazt_lab.features.a.public import main\nmain()\n", {"shape"}),
    ("from bazt_lab.features.b.public import main\nmain()\n", {"shape"}),
    ("def (:\n", {"syntax", "shape"}),
])
def test_a_feature_main_that_does_more_than_call_public_is_refused(source, expected):
    assert planted("bazt_lab/features/a/__main__.py", source) == expected


def test_other_root_modules_stay_refused_next_to_a_main():
    files = tree(extra={"bazt_lab/features/a/__main__.py": THIN_MAIN, "bazt_lab/features/a/cli.py": THIN_MAIN})
    assert [v.path for v in gate.check_sources(files)] == ["bazt_lab/features/a/cli.py"]
    assert planted("bazt_lab/__main__.py", THIN_MAIN) == {"shape"}


# -------------------------------------------------------------------------------------------------- secrets
TOKEN = "hf_" + "A1b2C3d4" * 4
OPENROUTER = "sk-or-v1-" + "0123456789abcdef" * 3


@pytest.mark.parametrize("source", [
    f'x = "{TOKEN}"\n',
    f'headers = {{"auth": "{OPENROUTER}"}}\n',
    'HF_TOKEN = "abcdefghij1234567890"\n',
    'client = make(api_key="abcdefghij1234567890")\n',
    'cfg = {"wandb_api_key": "abcdefghij1234567890"}\n',
    'import os\nkey = os.environ["WANDB_API_KEY"]\n',
    'import os\nkey = os.environ.get("HF_TOKEN")\n',
    'import os\nkey = os.getenv("OPENROUTER_API_KEY")\n',
    'from os import environ\nkey = environ["HF_TOKEN"]\n',
    "from dotenv import load_dotenv\n",
    "import dotenv\n",
    'from bazt_lab.core import public as core\nkey = core.get(settings, "wandb.api_key")\n',
])
def test_secrets_stay_in_the_environment_and_in_adapters(source):
    assert planted("bazt_lab/features/a/private/x.py", source) == {"secrets"}


@pytest.mark.parametrize("source", [
    'import os\nfrom dotenv import load_dotenv\nload_dotenv()\nkey = os.environ.get("HF_TOKEN")\n',
    'import os\nkey = os.getenv("OPENROUTER_API_KEY")\n',
    "from dotenv import load_dotenv\nload_dotenv(override=False)\n",
])
def test_adapters_read_secrets_from_the_environment(source):
    assert planted("bazt_lab/features/a/adapters/x.py", source) == set()


@pytest.mark.parametrize("call", ["load_dotenv(override=True)", "load_dotenv('.env', True and True, override=True)",
                                  "dotenv.load_dotenv(override=True)"])
def test_dot_env_never_overrides_the_environment(call):
    source = f"import dotenv\nfrom dotenv import load_dotenv\n{call}\n"
    assert planted("bazt_lab/features/a/adapters/x.py", source) == {"secrets"}


def test_a_secret_literal_is_refused_even_in_adapters():
    assert planted("bazt_lab/features/a/adapters/x.py", f'x = "{TOKEN}"\n') == {"secrets"}


@pytest.mark.parametrize("source", [
    'PAD_TOKEN = "<pad>"\n',
    'token = ""\n',
    'api_key = "OPENROUTER_API_KEY"\n',
    'max_tokens = 150\n',
    'input_tokens_per_agent_call = "mid estimate"\n',
])
def test_ordinary_token_words_are_not_secrets(source):
    assert planted("bazt_lab/features/a/private/x.py", source) == set()


def test_settings_file_holds_no_secret():
    assert rules(tree(extra={"experiments/profiles.toml": '[tracking]\nwandb_api_key = "x"\n'})) == {"secrets"}
    assert rules(tree(extra={"experiments/profiles.toml": f'[x]\nnote = "{TOKEN}"\n'})) == {"secrets"}
    assert rules(tree(extra={"experiments/profiles.toml": '[llm]\nmax_tokens = 3\n'})) == set()
    assert rules(tree(extra={"experiments/profiles.toml": "not toml ["})) == {"syntax"}


# ------------------------------------------------------------------------------------ settings, no defaults
@pytest.mark.parametrize("source", [
    'x = settings.get("llm.agents", 12)\n',
    'x = settings.get("llm.agents")\n',
    'def f(ctx):\n    return ctx.settings.get("llm.agents", 12)\n',
    'def f(s):\n    return s.get("llm.agents", 12)\n',
    'cfg = ctx.settings\nx = cfg.get("mock.batch_size")\n',
])
def test_settings_are_read_with_core_get_only(source):
    assert planted("bazt_lab/features/a/private/x.py", source) == {"settings-default"}


def test_core_get_is_the_way_to_read_settings():
    source = ('from bazt_lab.core import public as core\nx = core.get(settings, "llm.agents")\n'
              'y = {}.get("k", 1)\nz = row.get("unit")\n')
    assert planted("bazt_lab/features/a/private/x.py", source) == set()


# ------------------------------------------------------------------------------------------- reporting
def test_a_syntax_error_is_a_violation():
    assert planted("bazt_lab/features/a/private/x.py", "def (:\n") == {"syntax"}


def test_violations_name_file_line_and_rule():
    (v,) = gate.check_sources(tree(extra={"bazt_lab/features/a/private/x.py": "import torch\n\nimport wandb\n"}))
    assert (v.path, v.line, v.rule) == ("bazt_lab/features/a/private/x.py", 3, "service-adapter")
    assert str(v).startswith("bazt_lab/features/a/private/x.py:3: [service-adapter] ")


def test_violations_are_sorted_and_not_repeated():
    files = tree(extra={"bazt_lab/features/b/private/z.py": "import wandb\nimport wandb\n",
                        "bazt_lab/features/a/private/y.py": "import openai\n"})
    found = gate.check_sources(files)
    assert found == sorted(found) and len(found) == 3


# ---------------------------------------------------------------------------------------- disk and command line
def test_check_reads_a_tree_from_disk_and_skips_caches(tmp_path):
    root = write(tmp_path, tree(extra={"bazt_lab/features/a/__pycache__/x.cpython-311.pyc": "",
                                       "bazt_lab/__pycache__/y.pyc": "",
                                       "bazt_lab/features/ghost/__pycache__/z.pyc": ""}))
    assert gate.check(root) == []


def test_check_reads_the_settings_file_and_binary_data(tmp_path):
    root = write(tmp_path, tree(extra={"experiments/profiles.toml": '[tracking]\nwandb_api_key = "x"\n'}))
    (root / "bazt_lab/features/a/data.bin").write_bytes(b"\xff\xfe\x00\x01")
    assert {v.rule for v in gate.check(root)} == {"secrets"}


def test_check_finds_violations_in_files_on_disk(tmp_path):
    root = write(tmp_path, tree(extra={"bazt_lab/features/a/private/x.py": "import wandb\n"}))
    assert [v.rule for v in gate.check(root)] == ["service-adapter"]


def test_cli_exit_code_and_output(tmp_path, capsys):
    clean = write(tmp_path / "clean", tree())
    assert gate.main(["--root", str(clean)]) == 0
    dirty = write(tmp_path / "dirty", tree(extra={"bazt_lab/features/a/private/x.py": "import wandb\n"}))
    assert gate.main(["--root", str(dirty)]) == 1
    out = capsys.readouterr().out
    assert "bazt_lab/features/a/private/x.py:1: [service-adapter]" in out and "1 violation" in out


# ----------------------------------------------------------------------------------------------- real tree
def test_the_real_tree_passes_the_gate():
    found = gate.check(REPO)
    assert found == [], "\n".join(map(str, found))


# ------------------------------------------------------------------------------------- the composition root wires it all
def real_features() -> list[str]:
    root = REPO / "bazt_lab" / "features"
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "__init__.py").exists())


def test_every_feature_has_an_importable_public_module():
    import importlib
    missing = []
    for name in real_features():
        try:
            importlib.import_module(f"bazt_lab.features.{name}.public")
        except ImportError as error:
            missing.append(f"{name}: {error}")
    assert missing == []


def test_app_wires_every_arm_feature_both_platforms_the_tracker_and_sync():
    import importlib

    from bazt_lab import app
    from bazt_lab.core import public as core

    exporters = {name for name in real_features()
                 if hasattr(importlib.import_module(f"bazt_lab.features.{name}.public"), "ARMS")}
    assert exporters == set(app.ARM_FEATURES), f"app.ARM_FEATURES must list exactly the features that export ARMS: {exporters}"
    names = []
    for feature in app.ARM_FEATURES:
        arms = importlib.import_module(f"bazt_lab.features.{feature}.public").ARMS
        assert arms and all(isinstance(spec, core.ArmSpec) for spec in arms), feature
        names += [spec.name for spec in arms]
    assert len(names) == len(set(names)), "arm names are unique across features"
    assert set(app.PLATFORMS) == {"mock", "llm"}
    for platform, (feature, attribute) in app.PLATFORMS.items():
        assert hasattr(importlib.import_module(f"bazt_lab.features.{feature}.public"), attribute), platform
    feature, attribute = app.TRACKER
    assert hasattr(importlib.import_module(f"bazt_lab.features.{feature}.public"), attribute)
    source = (REPO / "bazt_lab" / "app.py").read_text(encoding="utf-8")
    assert "sync" in source, "app.py must wire the sync feature"
    assert "sync" in real_features()


def test_the_ruff_step_of_check_static_is_green_on_the_real_tree():
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-m", "ruff", "check", "--output-format", "concise"], cwd=REPO,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-500:]
