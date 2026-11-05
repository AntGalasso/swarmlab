"""Settings: experiments/profiles.toml with the chosen profile and `key=value` overrides applied.

No module in bazt_lab has a default for an experiment value. A missing key raises `Unset`, which names it.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = ROOT / "experiments" / "profiles.toml"


class Unset(KeyError):
    """A setting the caller needs is not set. `args[0]` is its dotted key."""


def put(tree: dict, dotted: str, value: Any) -> None:
    *path, leaf = dotted.split(".")
    for part in path:
        tree = tree.setdefault(part, {})
    tree[leaf] = value


def parse_value(text: str) -> Any:
    """A command-line value parsed as a TOML value (`80`, `[1, 2]`, `"x"`); bare words stay strings."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def load_settings(path: Path = SETTINGS_PATH, profile: str | None = None,
                  overrides: Iterable[str] = ()) -> dict:
    """profiles.toml, then the profile's dotted keys, then each `key=value` override. Adds `profile`."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    name = profile or raw.get("active_profile")
    profiles = raw.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"unknown profile {name!r}; {path} defines {sorted(profiles)}")
    settings = {k: v for k, v in raw.items() if k not in ("profiles", "active_profile")}
    for key, value in profiles[name].items():
        put(settings, key, value)
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {item!r}")
        put(settings, key.strip(), parse_value(text.strip()))
    settings["profile"] = name
    return settings


def get(settings: Mapping, dotted: str) -> Any:
    node: Any = settings
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise Unset(dotted)
        node = node[part]
    return node


def is_set(settings: Mapping, dotted: str) -> bool:
    try:
        get(settings, dotted)
    except Unset:
        return False
    return True
