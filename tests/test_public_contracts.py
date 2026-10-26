import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from src import main
from src.base_agent import BaseAgent
from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.config_loader import PROJECT_ROOT, load_dataset_plan, load_ollama_config
from src.deterministic_feedback import repair_scope
from src.export_agent import ExportAgent
from src.judge_agent import JudgeAgent
from src.knowledge_base import KnowledgeBase
from src.models import JudgeResult, JUDGE_DOMAIN_KEYS
from src.review_candidate_verifier import parse_candidates, verify_candidates
from src.run_validated_silver import input_paths, require_cuda


def test_no_ck_fails_before_any_output_or_model_call(monkeypatch, tmp_path):
    monkeypatch.delenv('GAMA_PCA_KNOWLEDGE_DIR', raising=False)
    with pytest.raises(FileNotFoundError, match='separate dataset'):
        BatchOrchestratorAgent(output_dir=tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_incomplete_external_ck_is_not_silently_accepted(tmp_path):
    with pytest.raises(FileNotFoundError, match='Incomplete external CK'):
        KnowledgeBase(knowledge_dir=tmp_path)


@pytest.mark.parametrize('package_root', [False, True])
def test_external_ck_contract_and_check_do_not_generate(tmp_path, monkeypatch, capsys, package_root):
    plan = load_dataset_plan()
    mappings = {name: {'schema_test': True} for name in KnowledgeBase.REQUIRED_FILES}
    mappings['archetypes'] = {'archetypes': {name: {} for name in plan['quotas_by_archetype']}}
    mappings['risk_groups'] = {'risk_groups': {'test_group': {}}}
    mappings['gleason_isup_mapping'] = {'gleason_to_isup': {'test_grade': 1}}
    mappings['treatment_options'] = {'allowed_treatments_by_risk_group': {'test_group': {}}}
    directory = tmp_path / 'clinical_knowledge' if package_root else tmp_path
    directory.mkdir(exist_ok=True)
    for name, value in mappings.items():
        (directory / (name + '.yaml')).write_text(yaml.safe_dump(value))
    assert KnowledgeBase(knowledge_dir=tmp_path).knowledge_dir == directory
    monkeypatch.setattr('sys.argv', ['gama-pca', '--knowledge-dir', str(tmp_path), '--check'])
    main.main()
    assert 'no model calls' in capsys.readouterr().out
    assert not (tmp_path / 'runtime').exists()


@pytest.mark.parametrize('error,scope', [
    ('patient_schema: age: invalid', 'patient'),
    ('psa_density_consistency: mismatch', 'diagnosis'),
    ('treatment_allowed_by_risk_group: mismatch', 'treatment'),
    ('events_sorted_by_month', 'progression'),
    ('unknown_rule: mismatch', None),
])
def test_partial_repair_ownership(error, scope):
    assert repair_scope([error]) == scope


def judge(scores, *, critical=False, revision=False):
    return JudgeResult(scores={name: {'score': score, 'comment': 'test'}
                              for name, score in zip(JUDGE_DOMAIN_KEYS, scores)},
                       critical_error=critical, clinically_acceptable=not (critical or revision),
                       requires_revision=revision, general_observations='test')


@pytest.mark.parametrize('scores,critical,revision,stage', [
    ([4] * 5 + [5] * 5, False, False, None),
    ([4] * 6 + [5] * 4, False, False, 'judge_mean_below_threshold'),
    ([5] * 10, True, False, 'judge_critical_error'),
    ([5] * 10, False, True, None),
])
def test_narrative_acceptance_separates_revision_flag(scores, critical, revision, stage):
    orch = object.__new__(BatchOrchestratorAgent)
    orch.dataset_plan = {'judge_minimum_mean_score': 4.5, 'reject_if_judge_critical_error': True}
    result = orch._narrative_rejection(None, judge(scores, critical=critical, revision=revision), 'text')
    assert (result['stage'] if result else None) == stage


def test_judge_prompt_is_narrative_only():
    agent = JudgeAgent('judge', 'mock', None, 'prompts/clinical_rubric_judge_parametric_prompt_es.txt')
    case = Mock()
    messages = agent.build_prompt(case, 'NARRATIVE_SENTINEL')
    assert 'NARRATIVE_SENTINEL' in messages[-1]['content']
    case.model_dump.assert_not_called()


def test_candidate_contract_rejects_verdict_or_extra_text():
    assert parse_candidates('{"candidates":[]}') == []
    for text in ('{"passed":true}', '{"candidates":[]} {"candidates":[]}'):
        with pytest.raises(ValueError):
            parse_candidates(text)


def test_unsupported_evidence_is_not_a_clinical_rejection():
    candidate = dict(case_path='missing', rule_path='unknown.yaml:missing',
                     yaml_value=1, case_value=2, condition='test')
    review, audit = verify_candidates([candidate], {}, {})
    assert review.passed and not review.errors
    assert audit['unsupported'] == 1


def test_export_does_not_rewrite_a_revision_flagged_narrative(tmp_path):
    exporter = ExportAgent(tmp_path)
    case = SimpleNamespace(case_id='test_note', model_dump=lambda: {'test': True})
    metadata = SimpleNamespace(model_dump=lambda: {'case_id': 'test_note'})
    exporter.export_case(case, 'ORIGINAL_TEXT', 'prompt', metadata, None,
                         judge([5] * 10, revision=True))
    assert (tmp_path / 'generated_txt/test_note.txt').read_text() == 'ORIGINAL_TEXT'
    request = json.loads((tmp_path / 'human_review_queue/test_note_review.json').read_text())
    assert request['status'] == 'pending' and request['narrative'] == 'ORIGINAL_TEXT'


def test_resource_configuration_and_prompts_are_packaged():
    config = load_ollama_config()['ollama']
    assert config['generation_model'] == 'glm-5.2:cloud'
    assert config['num_predict_validation'] == 6000
    assert sum(load_dataset_plan()['quotas_by_archetype'].values()) == 500
    for path in (PROJECT_ROOT / 'prompts').glob('*.txt'):
        assert BaseAgent('test', None, prompt_template_path=path).prompt_template()


def test_cuda_absence_is_explicit(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match='CUDA is required'):
        require_cuda()


def test_ner_input_is_independent_of_ck(tmp_path):
    (tmp_path / 'note_1.txt').write_text('test')
    assert [p.stem for p in input_paths(tmp_path)] == ['note_1']


def test_empty_ner_input_rejected(tmp_path):
    with pytest.raises(ValueError, match='no .txt'):
        input_paths(tmp_path)
