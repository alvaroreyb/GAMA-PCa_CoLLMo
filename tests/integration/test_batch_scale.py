import json
from unittest.mock import Mock

import pytest

from src.batch_orchestrator_agent import BatchOrchestratorAgent, RepeatedValidationError
from src.models import ValidationResult
from src.export_agent import ExportAgent
from src.ollama_client import OllamaClient


def test_cloud_calls_skip_local_pause(monkeypatch):
    response = Mock()
    response.json.return_value = {"message": {"content": "ok"}}
    monkeypatch.setattr("src.ollama_client.requests.post", Mock(return_value=response))
    sleep = Mock()
    monkeypatch.setattr("src.ollama_client.time.sleep", sleep)
    client = OllamaClient("http://localhost", sleep_after_call_seconds=15)
    for model in ("glm-5.2:cloud", "kimi-k2.6:cloud", "test-cloud-model-cloud"):
        assert client.chat(model, [], 0, 10) == "ok"
    sleep.assert_not_called()
    client.chat("test-local-model", [], 0, 10)
    sleep.assert_called_once_with(15)


def test_exhausted_cases_are_recorded_and_ids_not_reused(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch.dataset_plan["max_attempts_per_case"] = 2
    monkeypatch.setattr(orch, "_build_case", Mock(side_effect=RuntimeError("test failure")))
    assert orch.run(total_cases=2) == []
    report = json.loads(orch.batch_report_path.read_text())
    assert report["status"] == "completed"
    assert report["processed"] == report["rejected"] == 2
    assert report["accepted"] == 0
    assert [x["case_id"] for x in report["outcomes"]] == [
        "caso_sintetico_2300", "caso_sintetico_2301"
    ]
    assert all(x["attempts"] == 2 for x in report["outcomes"])
    with pytest.raises(FileExistsError):
        orch.run(total_cases=2)


def test_quota_overflow_and_existing_case_fail_before_generation(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    build = Mock()
    monkeypatch.setattr(orch, "_build_case", build)
    with pytest.raises(ValueError):
        orch.run(total_cases=501)
    (tmp_path / "generated_json" / "caso_sintetico_2300.json").touch()
    with pytest.raises(FileExistsError):
        orch.run(total_cases=1)
    build.assert_not_called()


def test_empty_interrupted_report_can_restart_without_losing_history(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch.dataset_plan["max_attempts_per_case"] = 1
    report_path = tmp_path / "batch_2300_2300.json"
    empty = {"status": "running", "planned": 1, "processed": 0,
             "accepted": 0, "rejected": 0, "outcomes": []}
    report_path.write_text(json.dumps(empty))
    monkeypatch.setattr(orch, "_build_case", Mock(side_effect=RuntimeError("test failure")))
    orch.run(total_cases=1)
    archived = list((tmp_path / "batch_history").glob("*.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text()) == empty
    assert json.loads(report_path.read_text())["processed"] == 1


def test_resume_keeps_completed_positions_and_original_artifacts(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch.dataset_plan["max_attempts_per_case"] = 1
    sequence = orch._archetype_sequence(2)
    first = {"case_id": "caso_sintetico_2300", "archetype": sequence[0],
             "status": "rejected", "attempts": 10}
    report_path = tmp_path / "batch_2300_2301.json"
    report_path.write_text(json.dumps({"status": "running", "planned": 2,
                                     "processed": 1, "accepted": 0, "rejected": 1, "outcomes": [first]}))
    artifact = tmp_path / "generated_artifacts/caso_sintetico_2301_attempt_01_00_patient.json"
    artifact.write_text("original")
    build = Mock(side_effect=RuntimeError("test failure"))
    monkeypatch.setattr(orch, "_build_case", build)
    orch.run(total_cases=2, resume=True)
    assert build.call_args.args[0] == "caso_sintetico_2301"
    report = json.loads(report_path.read_text())
    assert report["outcomes"][0] == first
    assert report["processed"] == 2
    assert report["rejected"] == 2
    assert artifact.read_text() == "original"


def test_repeated_validation_restarts_once_then_records_rejection(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    archetype = orch._archetype_sequence(1)[0]
    case = orch._build_case("caso_sintetico_2300", archetype)
    build = Mock(return_value=case)
    monkeypatch.setattr(orch, "_build_case", build)
    monkeypatch.setattr(orch.validator, "run", Mock(return_value=ValidationResult(
        passed=False, errors=["archetype_required_events: missing marker"])))
    assert orch.run(total_cases=1) == []
    assert build.call_count == 2
    assert len(list((tmp_path / "generated_artifacts").glob("*00_retry_scope.json"))) == 2
    report = json.loads(orch.batch_report_path.read_text())
    assert report["status"] == "completed"
    assert report["processed"] == report['rejected'] == 1
    assert report['outcomes'][0]['attempts'] == 4
    assert report['outcomes'][0]['rejection_reason'] == 'full_restart_validation_failed'
    assert orch._validation_feedback == ["archetype_required_events: missing marker"]


def test_resume_preserves_accepted_case_and_restores_totals(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch.run(total_cases=1)
    previous = json.loads(orch.batch_report_path.read_text())
    previous["planned"] = 2
    previous["status"] = "running"
    (tmp_path / "batch_2300_2301.json").write_text(json.dumps(previous))
    original_path = tmp_path / "generated_json/caso_sintetico_2300.json"
    original = original_path.read_bytes()
    cases = orch.run(total_cases=2, resume=True)
    report = json.loads(orch.batch_report_path.read_text())
    assert len(cases) == report["accepted"] == report["processed"] == 2
    assert original_path.read_bytes() == original
    assert report["outcomes"][1]["case_id"] == "caso_sintetico_2301"
