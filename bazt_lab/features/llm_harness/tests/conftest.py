import pytest

from bazt_lab.features.llm_harness.adapters import upstream


@pytest.fixture(scope="session")
def patched_root(tmp_path_factory):
    """The pinned checkout with the patch set applied, built once per test session."""
    return upstream.prepare(tmp_path_factory.mktemp("gsafeguard") / "patched")


@pytest.fixture(scope="session")
def families(patched_root):
    return {f: upstream.load_family(patched_root, f) for f in upstream.FAMILIES}
