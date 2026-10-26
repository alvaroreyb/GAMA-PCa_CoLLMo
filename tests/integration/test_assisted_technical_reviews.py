import json
from unittest.mock import Mock

import pytest

from src.batch_orchestrator_agent import BatchOrchestratorAgent, KimiReviewIncompleteError
from src.export_agent import ExportAgent
from src.resolve_technical_reviews import apply_decisions, case_digest, complete_candidate_prefix


def setup_case(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    monkeypatch.setattr(orch, '_run_kimi', Mock(side_effect=KimiReviewIncompleteError('length', '{"candidates":[')))
    orch.run(1)
    path = tmp_path / 'technical_review_queue/caso_sintetico_2300_incomplete_review.json'
    payload = json.loads(path.read_text())
    entry = dict(case_id=payload['case_id'], case_sha256=case_digest(payload['case']),
                 decision='approve', reason='Explicit assisted CK audit of saved skeleton')
    return orch, path, entry


def test_approved_technical_review_resumes_at_narrative_without_fake_kimi_approval(tmp_path, monkeypatch):
    orch, path, entry = setup_case(tmp_path, monkeypatch)
    assert apply_decisions(orch, [entry], apply=False) == [('caso_sintetico_2300', 'ready_for_narrative')]
    assert json.loads(path.read_text())['status'] == 'incomplete_review'
    apply_decisions(orch, [entry], apply=True)
    build = Mock(side_effect=AssertionError('No regeneration'))
    review = Mock(side_effect=AssertionError('No Kimi call'))
    monkeypatch.setattr(orch, '_build_case', build)
    monkeypatch.setattr(orch, '_run_kimi', review)
    assert len(orch.run(1, resume=True)) == 1
    build.assert_not_called()
    review.assert_not_called()
    metadata = json.loads((tmp_path / 'generated_metadata/caso_sintetico_2300.meta.json').read_text())
    assert metadata['kimi_review_passed'] is None
    assert not (tmp_path / 'generated_metadata/caso_sintetico_2300_kimi_review.json').exists()
    saved = json.loads((tmp_path / 'generated_json/caso_sintetico_2300.json').read_text())
    assert saved['metadata']['technical_review_resolution']['reviewer'] == 'human'
    assert json.loads(path.read_text())['status'] == 'resolved_automatically'


def test_doubts_stay_incomplete_without_export_or_new_model_call(tmp_path, monkeypatch):
    orch, path, entry = setup_case(tmp_path, monkeypatch)
    entry['decision'] = 'hold'
    apply_decisions(orch, [entry], apply=True)
    assert json.loads(path.read_text())['status'] == 'awaiting_user'
    report = json.loads(orch.batch_report_path.read_text())
    assert report['accepted'] == 0 and report['incomplete_review'] == 1
    assert report['awaiting_narrative'] == 0


def test_modified_skeleton_does_not_receive_a_stale_approval(tmp_path, monkeypatch):
    orch, path, entry = setup_case(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    payload['case']['patient']['age'] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='cambió'):
        apply_decisions(orch, [entry], apply=True)
    assert json.loads(path.read_text())['status'] == 'incomplete_review'


def test_prefix_extraction_does_not_repair_unfinished_object():
    raw = '{"candidates":[{"case_path":"one"},{"case_path":"two"},{"case_path":'
    assert complete_candidate_prefix(raw) == [{'case_path': 'one'}, {'case_path': 'two'}]


def test_pending_only_recovers_checkpoint_without_opening_new_positions(tmp_path, monkeypatch):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    monkeypatch.setattr(orch, '_run_kimi', Mock(side_effect=KimiReviewIncompleteError('length', '{"candidates":[')))
    orch.run(2, max_new_positions=1)
    path = tmp_path / 'technical_review_queue/caso_sintetico_2300_incomplete_review.json'
    payload = json.loads(path.read_text())
    apply_decisions(orch, [dict(case_id=payload['case_id'], case_sha256=case_digest(payload['case']),
                               decision='approve', reason='Independent CK audit')], apply=True)
    monkeypatch.setattr(orch, '_build_case', Mock(side_effect=AssertionError('No new case')))
    monkeypatch.setattr(orch, '_run_kimi', Mock(side_effect=AssertionError('No review call')))
    assert len(orch.run(2, resume=True, pending_only=True)) == 1
    report = json.loads(orch.batch_report_path.read_text())
    assert report['processed'] == 1 and report['status'] == 'paused_pending_only'
    assert not (tmp_path / 'generated_json/caso_sintetico_2301.json').exists()
    assert len(orch.run(2, resume=True, pending_only=True)) == 1
    with pytest.raises(ValueError, match='requires resume'):
        orch.run(2, pending_only=True)
