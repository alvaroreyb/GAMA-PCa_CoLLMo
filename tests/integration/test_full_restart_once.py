import json
from unittest.mock import Mock

from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.export_agent import ExportAgent
from src.models import ValidationResult


ERROR = 'archetype_required_events: missing marker'


def fixture(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    return orch


def test_full_restart_can_succeed_and_clears_all_partial_state(tmp_path, monkeypatch):
    orch = fixture(tmp_path)
    orch.dataset_plan['max_attempts_per_case'] = 3
    archetype = orch._archetype_sequence(1)[0]
    case = orch._build_case('caso_sintetico_2300', archetype)
    calls = []
    def build(*args):
        calls.append(args)
        if len(calls) == 2:
            assert orch._validation_feedback == []
            assert orch._treatment_retry is None
        return case.model_copy(deep=True)
    monkeypatch.setattr(orch, '_build_case', build)
    monkeypatch.setattr(orch.validator, 'run', Mock(side_effect=[
        ValidationResult(passed=False, errors=[ERROR]) for _ in range(3)
    ] + [ValidationResult(passed=True)]))
    assert len(orch.run(1)) == 1
    assert len(calls) == 2
    report = json.loads(orch.batch_report_path.read_text())
    assert report['accepted'] == 1 and report['outcomes'][0]['attempts'] == 4
    assert len(list((tmp_path / 'generated_artifacts').glob('*00_full_restart.json'))) == 1


def test_failed_full_restart_continues_to_next_id(tmp_path, monkeypatch):
    orch = fixture(tmp_path)
    build = Mock(wraps=orch._build_case)
    monkeypatch.setattr(orch, '_build_case', build)
    real_validate = orch.validator.run
    def validate(case):
        return ValidationResult(passed=False, errors=[ERROR]) if case.case_id == 'caso_sintetico_2300' else real_validate(case)
    monkeypatch.setattr(orch.validator, 'run', validate)
    assert len(orch.run(2)) == 1
    report = json.loads(orch.batch_report_path.read_text())
    assert [row['status'] for row in report['outcomes']] == ['rejected', 'accepted']
    assert report['processed'] == 2 and report['rejected'] == 1
    assert build.call_count == 3  # original + one full restart + next ID


def test_resume_old_repeated_error_pause_starts_fresh_immediately(tmp_path, monkeypatch):
    orch = fixture(tmp_path)
    archetype = orch._archetype_sequence(1)[0]
    case = orch._build_case('caso_sintetico_2300', archetype)
    (tmp_path / 'batch_2300_2300.json').write_text(json.dumps({
        'status': 'paused_repeated_validation_errors', 'planned': 1,
        'processed': 0, 'accepted': 0, 'rejected': 0, 'outcomes': [], 'sessions': []}))
    monkeypatch.setattr(orch, '_load_progression_retry', Mock(return_value=(case, [ERROR], 'old_candidate.json')))
    partial = Mock(side_effect=AssertionError('Must not pay for three more partial attempts'))
    monkeypatch.setattr(orch, '_regenerate_progression', partial)
    build = Mock(return_value=case)
    monkeypatch.setattr(orch, '_build_case', build)
    assert len(orch.run(1, resume=True)) == 1
    build.assert_called_once()
    partial.assert_not_called()


def test_resume_failed_full_restart_does_not_restart_again(tmp_path, monkeypatch):
    orch = fixture(tmp_path)
    archetype = orch._archetype_sequence(1)[0]
    case = orch._build_case('caso_sintetico_2300', archetype)
    orch.exporter.export_artifact(case.case_id, 'attempt_04_00_full_restart', {'scope': 'full_case_once'})
    (tmp_path / 'batch_2300_2300.json').write_text(json.dumps({
        'status': 'running', 'planned': 1, 'processed': 0, 'outcomes': [], 'sessions': []}))
    monkeypatch.setattr(orch, '_load_progression_retry', Mock(return_value=(case, [ERROR], 'failed_fresh.json')))
    build = Mock(side_effect=AssertionError('Must not pay for another restart'))
    monkeypatch.setattr(orch, '_build_case', build)
    assert orch.run(1, resume=True) == []
    build.assert_not_called()
    report = json.loads(orch.batch_report_path.read_text())
    assert report['rejected'] == 1 and report['status'] == 'completed'
