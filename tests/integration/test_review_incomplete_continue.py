import json
from unittest.mock import Mock

import pytest

from src.batch_orchestrator_agent import BatchOrchestratorAgent, KimiReviewIncompleteError
from src.export_agent import ExportAgent
from src.models import KimiReviewResult
from src.ollama_client import TruncatedResponseError


def test_incomplete_review_preserved_then_next_id_runs_without_regeneration(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    build = Mock(wraps=orch._build_case)
    monkeypatch.setattr(orch, '_build_case', build)
    review = Mock(side_effect=[KimiReviewIncompleteError('truncated', '{"candidates":['), KimiReviewResult(passed=True)])
    monkeypatch.setattr(orch, '_run_kimi', review)
    assert len(orch.run(2)) == 1
    assert build.call_count == review.call_count == 2
    report = json.loads(orch.batch_report_path.read_text())
    assert [row['status'] for row in report['outcomes']] == ['incomplete_review', 'accepted']
    assert report['incomplete_review'] == 1 and report['rejected'] == 0
    path = tmp_path / 'technical_review_queue/caso_sintetico_2300_incomplete_review.json'
    payload = json.loads(path.read_text())
    assert payload['raw_response'] == '{"candidates":['
    assert payload['clinical_verdict'] is None and not payload['automatic_retry']
    assert not (tmp_path / 'generated_json/caso_sintetico_2300.json').exists()
    assert not list((tmp_path / 'human_review_queue').glob('*2300*'))
    original = path.read_bytes()
    build.reset_mock()
    review.reset_mock()
    assert len(orch.run(2, resume=True)) == 1
    build.assert_not_called()
    review.assert_not_called()
    assert path.read_bytes() == original


def test_exact_logged_truncation_is_not_paid_again_on_resume(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    case = orch._build_case('caso_sintetico_2300', orch._archetype_sequence(1)[0])
    orch.app_config['app']['use_llm_generation'] = True
    agent = orch.kimi_agent
    agent.candidate_review = True
    client = Mock()
    client.trace_dir = tmp_path
    agent.ollama_client = client
    def chat(model, messages, temperature, budget, **kwargs):
        (tmp_path / '20261008_kimi_case_review.json').write_text(json.dumps({
            'model': model, 'messages': messages, 'temperature': temperature, 'num_predict': budget,
            'attempts': [{'response': {'done_reason': 'length',
                'message': {'content': '{"candidates":['}}}]}))
        raise TruncatedResponseError(model, '{"candidates":[')
    client.chat.side_effect = chat
    with pytest.raises(KimiReviewIncompleteError):
        orch._run_kimi(case)
    agent.reuse_logged_response = True
    with pytest.raises(KimiReviewIncompleteError) as exc:
        orch._run_kimi(case)
    assert exc.value.raw_response == '{"candidates":['
    client.chat.assert_called_once()


def test_invalid_json_is_incomplete_not_a_clinical_rejection(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    case = orch._build_case('caso_sintetico_2300', orch._archetype_sequence(1)[0])
    orch.app_config['app']['use_llm_generation'] = True
    orch.kimi_agent.candidate_review = True
    orch.kimi_agent.ollama_client = Mock()
    orch.kimi_agent.ollama_client.chat.return_value = 'not JSON'
    with pytest.raises(KimiReviewIncompleteError) as exc:
        orch._run_kimi(case)
    assert exc.value.raw_response == 'not JSON'
    orch.kimi_agent.ollama_client.chat.assert_called_once()
