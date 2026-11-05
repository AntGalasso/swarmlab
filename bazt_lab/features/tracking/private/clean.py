"""What leaves the process: the run config, table cells, artifact names. Pure functions."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from bazt_lab.features.tracking.private.errors import TrackingError

# Same shape as the architecture gate's secret-name rule: the last word of a key says it holds a credential.
# camelCase and hyphenated spellings are read as words first (accessToken, api-key).
SECRET_NAME = re.compile(r"(?:^|_)(?:token|api_?key|secret|password)$")
WORD_BREAK = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[-\s]+")


def is_secret_name(key: str) -> bool:
    return bool(SECRET_NAME.search(WORD_BREAK.sub("_", key).lower()))
NOT_ALLOWED = re.compile(r"[^A-Za-z0-9_.-]")


def clean_config(config: Mapping) -> dict:
    """The resolved settings as a W&B config: plain JSON, no secret-named key at any depth, no period in any key
    (W&B config keys must not contain periods; "." becomes "_" and two keys that would merge are refused)."""
    return _clean(json.loads(json.dumps(config, default=_plain)), "config")


def _plain(value: Any) -> Any:
    """json.dumps fallback: numpy and torch values keep their numbers, anything else becomes its text."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def _clean(node: Any, where: str) -> Any:
    if isinstance(node, list):
        return [_clean(item, where) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict = {}
    for key, value in node.items():
        if is_secret_name(key):
            continue
        safe = key.replace(".", "_")
        if safe in out:
            raise TrackingError(f"{where}: two settings keys become {safe!r} once their periods are replaced "
                                "(W&B config keys cannot contain periods); rename one of them")
        out[safe] = _clean(value, f"{where}.{safe}")
    return out


def cell(value: Any) -> Any:
    """A table cell: None, bool, int, float (NaN and infinity stay) and str are kept as exact Python types; numpy
    and torch scalars become Python scalars; anything else (lists, dicts, arrays, objects) becomes JSON text."""
    if value is None:
        return None
    for plain in (bool, int, float, str):
        if isinstance(value, plain):
            return plain(value)
    if hasattr(value, "tolist"):                              # numpy and torch scalars and arrays
        value = value.tolist()
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
    return json.dumps(value, default=_plain)


def wandb_name(text: str) -> str:
    """An artifact name or type W&B accepts: letters, digits, underscore, hyphen and dot."""
    return NOT_ALLOWED.sub("_", text)
