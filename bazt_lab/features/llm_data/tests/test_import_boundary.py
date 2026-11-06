"""This feature's own slice of the architecture rules (docs/ARCHITECTURE_EXPERIMENTS.md, import rules 1-3)."""
import ast
from pathlib import Path

FEATURE = Path(__file__).resolve().parents[1]
THIRD_PARTY_SERVICES = {"wandb", "datasets", "huggingface_hub", "openai", "sentence_transformers",
                        "torch_geometric", "swarmlab", "bazt", "llm_client"}


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield from (f"{node.module}.{a.name}" for a in node.names)


def sources(*, adapters):
    for path in FEATURE.rglob("*.py"):
        if "tests" not in path.relative_to(FEATURE).parts and (path.parent.name == "adapters") == adapters:
            yield path


def test_only_adapters_import_third_party_services():
    for path in sources(adapters=False):
        for mod in imported_modules(path):
            assert mod.split(".")[0] not in THIRD_PARTY_SERVICES, f"{path.name} imports {mod}"
    assert {p.name for p in sources(adapters=True)} >= {"hf.py"}


def test_other_features_and_core_are_reached_through_public_only():
    for path in sources(adapters=False):
        for mod in imported_modules(path):
            parts = mod.split(".")
            if parts[:2] == ["bazt_lab", "features"] and parts[2] != "llm_data":
                assert parts[3:4] in ([], ["public"]), f"{path.name} imports {mod}"
            if parts[:2] == ["bazt_lab", "core"]:
                assert parts[2:] == ["public"], f"{path.name} imports {mod}"
            assert parts[:2] != ["bazt_lab", "app"]
