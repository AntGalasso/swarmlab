"""What leaves the process: the run config, table cells, artifact names."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from bazt_lab.features.tracking.private.clean import cell, clean_config, wandb_name
from bazt_lab.features.tracking.private.errors import TrackingError


def test_config_drops_every_secret_named_key_at_every_depth():
    config = {"profile": "sprint", "WANDB_API_KEY": "x y", "llm": {"judge": {"api_key": "x y", "model": "m"},
              "hf_token": "x y", "Secret": "x y", "db_password": "x y"}, "keys": ["a"], "tokens": 3}
    got = clean_config(config)
    assert got == {"profile": "sprint", "llm": {"judge": {"model": "m"}}, "keys": ["a"], "tokens": 3}


@pytest.mark.parametrize("name", ["accessToken", "api-key", "apiKey", "API_KEY", "hf-token", "clientSecret",
                                  "dbPassword", "refresh_token"])
def test_config_drops_secret_names_written_in_camel_case_or_with_hyphens(name):
    assert clean_config({"keep": 1, "x": {name: "v"}}) == {"keep": 1, "x": {}}


@pytest.mark.parametrize("name", ["tokens", "keyboard", "monkey", "passwords", "max_tokens_per_call"])
def test_config_keeps_names_that_only_look_like_secrets(name):
    assert clean_config({name: 1}) == {name: 1}


def test_config_does_not_touch_its_input():
    config = {"a": {"b.c": 1, "api_key": "x y"}}
    clean_config(config)
    assert config == {"a": {"b.c": 1, "api_key": "x y"}}


def test_config_keys_lose_their_periods_because_wandb_forbids_them():
    config = {"estimates": {"usd": {"meta-llama/llama-3.3-70b-instruct": [0.1, 0.32]}}, "12": {"x.y": 1}}
    got = clean_config(config)
    assert got == {"estimates": {"usd": {"meta-llama/llama-3_3-70b-instruct": [0.1, 0.32]}}, "12": {"x_y": 1}}


def test_config_refuses_two_keys_that_would_merge():
    with pytest.raises(TrackingError, match="a_b"):
        clean_config({"a.b": 1, "a_b": 2})


def test_config_is_plain_json_whatever_it_was_given():
    config = {"path": Path("results/x"), "pair": (1, 2), "n": np.int64(3), "int_key": {7: "v"}}
    got = clean_config(config)
    assert got == {"path": str(Path("results/x")), "pair": [1, 2], "n": 3, "int_key": {"7": "v"}}
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("value", [None, True, False, 0, 3, -2, 1.5, "text", ""])
def test_cell_passes_plain_values_through_unchanged(value):
    got = cell(value)
    assert got == value and type(got) is type(value)


def test_cell_hands_wandb_exact_python_types_even_for_float_subclasses():
    assert type(cell(np.float64(2.5))) is float


def test_cell_keeps_nan_and_infinity_as_floats():
    assert math.isnan(cell(float("nan"))) and cell(float("inf")) == math.inf


@pytest.mark.parametrize("value, expected", [(np.float64(2.5), 2.5), (np.int64(4), 4), (np.bool_(True), True),
                                             (torch.tensor(1.5, dtype=torch.float64), 1.5), (torch.tensor(3), 3)])
def test_cell_turns_numpy_and_torch_scalars_into_python_scalars(value, expected):
    got = cell(value)
    assert got == expected and type(got) in (float, int, bool)


def test_cell_writes_lists_dicts_and_arrays_as_json_text():
    assert cell([1, 2.5, "a"]) == '[1, 2.5, "a"]'
    assert json.loads(cell({"a": (1, 2)})) == {"a": [1, 2]}
    assert json.loads(cell(torch.tensor([1.0, 2.0]))) == [1.0, 2.0]
    assert json.loads(cell(np.array([[1, 2], [3, 4]]))) == [[1, 2], [3, 4]]


def test_cell_never_fails_on_an_object_it_does_not_know():
    assert cell(Path("a/b")) == json.dumps(str(Path("a/b")))


@pytest.mark.parametrize("raw, expected", [("mock_main-manifest", "mock_main-manifest"), ("a b/c", "a_b_c"),
                                           ("v1.2", "v1.2"), ("rows:50%", "rows_50_")])
def test_wandb_names_keep_letters_digits_underscore_hyphen_dot(raw, expected):
    assert wandb_name(raw) == expected
