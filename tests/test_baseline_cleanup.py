import pytest

from scripts import baseline


def test_baseline_does_not_allocate_resources_without_sorry(monkeypatch, tmp_path):
    # Given
    allocations = []
    monkeypatch.setattr(
        baseline,
        "LLMClient",
        lambda **kwargs: allocations.append("client"),
    )
    monkeypatch.setattr(
        baseline,
        "LeanWorkDir",
        lambda project_dir: allocations.append("lean"),
    )

    # When
    result = baseline.run_baseline(
        name="complete",
        theorem_lean="example : True := by trivial",
        theorem_informal="",
        lean_project_dir=tmp_path,
        model="sonnet",
        run_dir=tmp_path / "run",
        max_tokens=1,
    )

    # Then
    assert result["status"] == "error"
    assert allocations == []


def test_baseline_cleans_client_before_lean_directory_on_interrupt(monkeypatch, tmp_path):
    # Given
    cleanup_order = []

    class ClientSpy:
        mistral = False

        def call(self, **kwargs):
            raise KeyboardInterrupt

        def cleanup(self):
            cleanup_order.append("client")

    class WorkDirSpy:
        def __init__(self, project_dir):
            pass

        def cleanup(self):
            cleanup_order.append("lean")

    monkeypatch.setattr(baseline, "LLMClient", lambda **kwargs: ClientSpy())
    monkeypatch.setattr(baseline, "LeanWorkDir", WorkDirSpy)

    # When
    with pytest.raises(KeyboardInterrupt):
        baseline.run_baseline(
            name="interrupted",
            theorem_lean="example : True := by\n  sorry",
            theorem_informal="",
            lean_project_dir=tmp_path,
            model="sonnet",
            run_dir=tmp_path / "run",
            max_tokens=1,
        )

    # Then
    assert cleanup_order == ["client", "lean"]
