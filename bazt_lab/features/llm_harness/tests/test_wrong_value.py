"""The attacker's wrong number exists for any gold format of GSM8K."""
import pytest

from bazt_lab.features.llm_data import public as data
from bazt_lab.features.llm_harness.private import cases


@pytest.mark.parametrize("gold", ["2.5", "18", "$18", "18.", "1,234", "0"])
def test_the_wrong_number_exists_for_any_gold_format(gold):
    item = data.TaskItem("gsm8k", "gsm8k:1", "How many?", (), gold, data.Source("x", None, None, None, None), {})
    wrong = cases.wrong_value(item, ("w", 1))
    assert wrong.isdigit() and float(wrong) > float(gold.replace(",", "").lstrip("$").rstrip("."))
