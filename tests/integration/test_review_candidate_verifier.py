import json
from unittest.mock import Mock

import pytest

from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.export_agent import ExportAgent
from src.review_candidate_verifier import parse_candidates, verify_candidates


@pytest.fixture
def sample(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    case = orch._build_case('candidate_test', 'unfavourable_intermediate_rp_biochemical_recurrence')
    case.patient.presentation_context = 'Biopsia Gleason 4+3 (Grupo de grado 2).'
    case.diagnosis.biopsy.gleason = '4+3'
    case.diagnosis.biopsy.isup = 3
    case.diagnosis.initial_psa = 12.5
    case.diagnosis.prostate_volume_cc = 50
    case.diagnosis.psa_density = 0.25
    context = json.loads(json.dumps(orch.kimi_agent._knowledge_context(case, orch.kb)))
    return orch, case, context


def candidate(path, rule, yaml, actual):
    return dict(case_path=path, rule_path=rule, yaml_value=yaml, case_value=actual,
                condition='Posible contradicción con la regla citada')


def density(context, actual=0.25):
    file = 'clinical_knowledge/hard_constraints.yaml'
    index = next(i for i, rule in enumerate(context[file]['hard_constraints']) if rule['id'] == 'psa_density_consistency')
    return candidate('diagnosis.psa_density', f'{file}:hard_constraints.{index}.tolerance_abs',
                     context[file]['hard_constraints'][index]['tolerance_abs'], actual)


def test_text_grade_real_contradiction_survives_correct_structured_isup(sample):
    _, case, context = sample
    item = candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, 2)
    review, audit = verify_candidates([item], case.model_dump(), context)
    assert not review.passed and len(review.errors) == 1
    assert audit['confirmed'] == 1 and audit['pending'] == 0
    assert 'ISUP documentado=2, YAML=3' in review.errors[0]


def test_density_within_yaml_tolerances_is_not_a_rejection(sample):
    _, case, context = sample
    case.diagnosis.psa_density = 0.26
    review, audit = verify_candidates([density(context, 0.26)], case.model_dump(), context)
    assert review.passed and review.errors == []
    assert audit['not_violated'] == 1


def test_density_outside_yaml_tolerances_is_confirmed(sample):
    _, case, context = sample
    case.diagnosis.psa_density = 0.5
    review, audit = verify_candidates([density(context, 0.5)], case.model_dump(), context)
    assert not review.passed and audit['confirmed'] == 1


def test_unknown_or_fabricated_evidence_does_not_block(sample):
    _, case, context = sample
    for item in (
        candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 2, 2),
        candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, 1),
        candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:missing', 3, 2),
        candidate('missing.path', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, 2),
    ):
        review, audit = verify_candidates([item], case.model_dump(), context)
        assert review.passed and not review.errors and audit['unsupported'] == 1


def test_correct_mapping_is_not_confirmed(sample):
    _, case, context = sample
    item = candidate('diagnosis.biopsy.isup', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, 3)
    assert verify_candidates([item], case.model_dump(), context)[0].passed


def test_unsupported_candidate_does_not_hide_a_confirmed_error(sample):
    _, case, context = sample
    real = candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, 2)
    invalid = candidate('missing', 'clinical_knowledge/gleason_isup_mapping.yaml:missing', 1, 1)
    review, audit = verify_candidates([invalid, real], case.model_dump(), context)
    assert not review.passed and len(review.errors) == 1
    assert audit['unsupported'] == 1 and audit['confirmed'] == 1 and audit['pending'] == 0


def test_null_without_explicit_prohibition_is_not_a_contradiction(sample):
    _, case, context = sample
    case.diagnosis.biopsy.tumour_extent = None
    file = 'clinical_knowledge/biopsy_rules.yaml'
    item = candidate('diagnosis.biopsy.tumour_extent', file + ':pathology_required_fields', context[file]['pathology_required_fields'], None)
    review, audit = verify_candidates([item], case.model_dump(), context)
    assert review.passed and audit['not_violated'] == 1


def test_inapplicable_risk_condition_is_not_a_rejection(sample):
    _, case, context = sample
    case.risk_classification.risk_group = 'metastatic_hormone_sensitive_at_diagnosis'
    file = 'clinical_knowledge/hard_constraints.yaml'
    index = next(i for i, rule in enumerate(context[file]['hard_constraints']) if rule['id'] == 'systemic_staging_before_committee_or_treatment_strong_groups')
    item = candidate('risk_classification.risk_group', f'{file}:hard_constraints.{index}.requires_previous_event',
                     'systemic_staging_before_treatment', case.risk_classification.risk_group)
    review, audit = verify_candidates([item], case.model_dump(), context)
    assert review.passed and audit['not_violated'] == 1


def test_satisfied_previous_event_is_not_a_rejection(sample):
    _, case, context = sample
    index = next(i for i, e in enumerate(case.progression.trajectory) if e.event_type == 'salvage_radiotherapy')
    file = 'clinical_knowledge/progression_rules.yaml'
    rule = context[file]['event_definitions']['salvage_radiotherapy']['required_previous_events_any']
    item = candidate(f'progression.trajectory.{index}.event_type', file + ':event_definitions.salvage_radiotherapy.required_previous_events_any', rule, 'salvage_radiotherapy')
    review, audit = verify_candidates([item], case.model_dump(), context)
    assert review.passed and audit['not_violated'] == 1


@pytest.mark.parametrize('raw', ['invalid', '{"passed":true,"errors":[]}', '{"candidates":[]} extra',
                               '{"candidates":[{}]}', '{"candidates":[],"passed":true}'])
def test_invalid_candidate_contract_is_not_repaired(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_candidates(raw)


def test_one_call_no_paid_repair_and_unsupported_does_not_pause(sample):
    orch, case, context = sample
    agent = orch.kimi_agent
    agent.candidate_review = True
    agent.cache_friendly_prefix = False
    agent.ollama_client = Mock()
    item = candidate('diagnosis.psa_density', 'clinical_knowledge/risk_groups.yaml:classification_order',
                     context['clinical_knowledge/risk_groups.yaml']['classification_order'], 0.25)
    agent.ollama_client.chat.return_value = json.dumps({'candidates': [item]})
    assert agent.run(case, orch.kb).passed
    agent.ollama_client.chat.assert_called_once()
    assert agent.ollama_client.chat.call_args.kwargs['max_attempts'] == 1
    assert agent.last_candidate_review['unsupported'] == 1


def test_no_candidates_returns_approval_with_one_call(sample):
    orch, case, _ = sample
    agent = orch.kimi_agent
    agent.candidate_review = True
    agent.ollama_client = Mock()
    agent.ollama_client.chat.return_value = '{"candidates":[]}'
    assert agent.run(case, orch.kb).passed
    agent.ollama_client.chat.assert_called_once()


def test_boolean_cannot_impersonate_numeric_grade(sample):
    _, case, context = sample
    item = candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, True)
    assert verify_candidates([item], case.model_dump(), context)[1]['unsupported'] == 1


@pytest.mark.parametrize('value', ['2', 'Grupo de grado 2', 'Grado 2 de ISUP', 'ISUP 2'])
def test_literal_grade_labels_can_be_verified_without_model_repair(sample, value):
    _, case, context = sample
    item = candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, value)
    assert verify_candidates([item], case.model_dump(), context)[1]['confirmed'] == 1


def test_free_prose_is_not_coerced_into_grade(sample):
    _, case, context = sample
    item = candidate('patient.presentation_context', 'clinical_knowledge/gleason_isup_mapping.yaml:gleason_to_isup.4+3', 3, 'debe tener ISUP 2 o 3')
    assert verify_candidates([item], case.model_dump(), context)[1]['unsupported'] == 1


def test_all_treatment_options_are_considered_not_only_cited_subset(sample):
    _, case, context = sample
    case.risk_classification.risk_group = 'low_risk'
    case.treatment_plan.initial_treatment = 'radical_prostatectomy'
    file = 'clinical_knowledge/treatment_options.yaml'
    item = candidate('treatment_plan.initial_treatment', file + ':allowed_treatments_by_risk_group.low_risk.preferred',
                     ['active_surveillance'], 'radical_prostatectomy')
    assert verify_candidates([item], case.model_dump(), context)[1]['not_violated'] == 1


def test_unsupported_time_anchor_does_not_block_closed_world_review(sample):
    _, case, context = sample
    index = next(i for i, e in enumerate(case.progression.trajectory) if e.event_type == 'salvage_radiotherapy')
    file = 'clinical_knowledge/progression_rules.yaml'
    rule = context[file]['event_definitions']['salvage_radiotherapy']['typical_month_range_after_previous']
    item = candidate(f'progression.trajectory.{index}.month', file + ':event_definitions.salvage_radiotherapy.typical_month_range_after_previous',
                     rule, case.progression.trajectory[index].month)
    review, audit = verify_candidates([item], case.model_dump(), context)
    assert review.passed and review.errors == [] and audit['unsupported'] == 1


def test_unsupported_audit_is_saved_without_orchestrator_pause(sample, tmp_path):
    orch, case, context = sample
    orch.app_config['app']['use_llm_generation'] = True
    orch.kimi_agent.candidate_review = True
    orch.kimi_agent.ollama_client = Mock()
    item = candidate('diagnosis.psa_density', 'clinical_knowledge/risk_groups.yaml:classification_order',
                     context['clinical_knowledge/risk_groups.yaml']['classification_order'], 0.25)
    orch.kimi_agent.ollama_client.chat.return_value = json.dumps({'candidates': [item]})
    assert orch._run_kimi(case).passed
    files = list(tmp_path.rglob('*03_candidate_verification.json'))
    assert len(files) == 1
    assert json.loads(files[0].read_text())['verification']['unsupported'] == 1
