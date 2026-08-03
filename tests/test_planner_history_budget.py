import re

import pytest

from openprover.prompts import format_planner_prompt


def _action_contents(prompt: str) -> list[str]:
    return re.findall(r"<action_output>\n(.*?)\n</action_output>", prompt, re.DOTALL)


def _prompt(*, outputs: list[str], history_budget: int) -> str:
    return format_planner_prompt(
        "whiteboard",
        "",
        [{"step": 1, "planner": "planner text", "outputs": [
            {"action": "act", "summary": "summary", "output": output}
            for output in outputs
        ]}],
        "budget",
        history_budget=history_budget,
    )


def test_single_output_keeps_full_output_allocation() -> None:
    contents = _action_contents(_prompt(outputs=["abcdefghij"], history_budget=15))

    assert [len(content) for content in contents] == [5]


def test_multiple_nonempty_outputs_share_output_allocation() -> None:
    contents = _action_contents(
        _prompt(outputs=["abcdefghij", "klmnopqrst"], history_budget=15),
    )

    assert [len(content) for content in contents] == [3, 2]


def test_empty_outputs_do_not_consume_a_share() -> None:
    contents = _action_contents(
        _prompt(outputs=["abcdefghij", "", "klmnopqrst"], history_budget=15),
    )

    assert [len(content) for content in contents] == [3, 2]


@pytest.mark.parametrize("history_budget", range(5))
def test_tiny_history_budgets_have_safe_output_lengths(history_budget: int) -> None:
    contents = _action_contents(_prompt(outputs=["abcdefghij"], history_budget=history_budget))

    output_limit = history_budget - history_budget * 2 // 3
    assert all(len(content) <= output_limit for content in contents)
