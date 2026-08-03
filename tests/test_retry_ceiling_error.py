from openprover.prover import MAX_CONSECUTIVE_ERRORS, Prover


def test_retry_ceiling_marks_final_llm_error_and_preserves_first_message():
    prover = Prover.__new__(Prover)
    prover._llm_error_exit = False
    prover._last_error_msg = "first failure"

    action = prover._retry_action(
        RuntimeError("latest failure"), MAX_CONSECUTIVE_ERRORS,
    )

    assert action == "stop"
    assert prover._llm_error_exit is True
    assert prover._last_error_msg == "first failure"


def test_retry_below_ceiling_keeps_policy_behavior():
    prover = Prover.__new__(Prover)
    prover._llm_error_exit = False
    prover._last_error_msg = ""
    prover._check_error_policy = lambda error: "retry"

    action = prover._retry_action(
        RuntimeError("retryable"), MAX_CONSECUTIVE_ERRORS - 1,
    )

    assert action == "retry"
    assert prover._llm_error_exit is False
    assert prover._last_error_msg == ""
