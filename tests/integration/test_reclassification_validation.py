from unittest.mock import Mock

import pytest

from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.deterministic_feedback import repair_scope
from src.export_agent import ExportAgent
from src.models import ClinicalEvent
from src.reclassification_validation import classify, validate_reclassifications


@pytest.fixture
def sample(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch._artifact_prefix = 'test_'
    case = orch._build_case('test_reclassification', 'low_risk_active_surveillance_progression_to_treatment')
    event = next(e for e in case.progression.trajectory if e.event_type == 'active_surveillance_reclassification')
    event.psa = 7.5
    event.notes = 'Biopsia de repetición Gleason 3+4 (ISUP 2). Reclasificación a riesgo intermedio favorable.'
    return orch, case, event


def test_correct_reclassification_inherits_tnm_without_repetition(sample):
    orch, case, _ = sample
    errors, warnings = validate_reclassifications(case, orch.kb)
    assert errors == [] and warnings == []


@pytest.mark.parametrize('psa', [6.3, 7.4, 7.5])
def test_wrong_unfavourable_label_has_precise_progression_feedback(sample, psa):
    orch, case, event = sample
    event.psa = psa
    event.notes = event.notes.replace('intermedio favorable', 'intermedio desfavorable')
    errors, _ = validate_reclassifications(case, orch.kb)
    assert len(errors) == 1
    error = errors[0]
    assert 'esperado=favourable_intermediate' in error
    assert 'declarado=unfavourable_intermediate' in error
    assert f'psa={psa}' in error and 'isup=2' in error
    assert 'cT1c' in error and 'diagnosis' in error
    assert repair_scope(errors) == 'progression'
    result = orch.validator.run(case)
    assert error in result.errors
    assert orch.validator.progression_only_failure(result)


@pytest.mark.parametrize('isup,psa,stage,expected', [
    (2, 9.99, 'cT1c', 'favourable_intermediate'),
    (2, 10.0, 'cT1c', 'unfavourable_intermediate'),
    (2, 20.0, 'cT1c', 'unfavourable_intermediate'),
    (2, 20.01, 'cT1c', 'high_risk_localised'),
    (3, 7.5, 'cT1c', 'unfavourable_intermediate'),
    (2, 7.5, 'cT3a', 'locally_advanced'),
])
def test_actual_yaml_all_any_thresholds_and_priority(sample, isup, psa, stage, expected):
    orch, _, _ = sample
    state = dict(isup=isup, psa=psa, cT=stage, cN='cN0', cM='cM0')
    assert classify(state, orch.kb.get('risk_groups')) == expected


def test_previous_text_tnm_update_is_used_without_repeating_it(sample):
    orch, case, event = sample
    prior = case.progression.trajectory[1]
    prior.notes = 'Nuevo estadio clínico documentado cT3a cN0 cM0.'
    event.notes = 'Biopsia ISUP 2. Nuevo riesgo: localmente avanzado.'
    assert validate_reclassifications(case, orch.kb)[0] == []
    event.notes = 'Biopsia ISUP 2. Riesgo intermedio favorable.'
    errors, _ = validate_reclassifications(case, orch.kb)
    assert len(errors) == 1 and 'esperado=locally_advanced' in errors[0]
    assert 'cT=cT3a (trajectory[1]' in errors[0]


def test_future_tnm_cannot_justify_reclassification(sample):
    orch, case, event = sample
    event.notes = 'ISUP 2. Reclasificación a riesgo intermedio desfavorable.'
    case.progression.trajectory.append(ClinicalEvent(month=100, event_type='follow_up', state='follow_up',
                                                   notes='Nuevo estadio cT3a cN1 cM1b.'))
    errors, _ = validate_reclassifications(case, orch.kb)
    assert len(errors) == 1 and 'esperado=favourable_intermediate' in errors[0]


def test_ambiguous_histology_is_not_a_false_rejection(sample):
    orch, case, event = sample
    event.notes = 'ISUP 1 inicialmente, ISUP 2 ahora. Riesgo intermedio desfavorable.'
    errors, warnings = validate_reclassifications(case, orch.kb)
    assert errors == []
    assert warnings and warnings[0].startswith('reclassification_risk_unverifiable:')


def test_gleason_alone_can_supply_grade_without_new_required_field(sample):
    orch, case, event = sample
    event.notes = 'Gleason 4+3. Reclasificación a riesgo intermedio desfavorable.'
    assert validate_reclassifications(case, orch.kb) == ([], [])


def test_missing_psa_uses_previous_documented_value(sample):
    orch, case, event = sample
    case.progression.trajectory[1].psa = 12.0
    event.psa = None
    event.notes = 'ISUP 2. Reclasificación a riesgo intermedio desfavorable.'
    assert validate_reclassifications(case, orch.kb) == ([], [])


def test_psa_in_notes_can_update_previous_value(sample):
    orch, case, event = sample
    event.psa = None
    event.notes = 'PSA de 12,5 ng/mL. ISUP 2. Riesgo intermedio desfavorable.'
    assert validate_reclassifications(case, orch.kb) == ([], [])


def test_compact_previous_clinical_tnm_is_recognized(sample):
    orch, case, event = sample
    case.progression.trajectory[1].notes = 'Estadio clínico cT3acN0cM0.'
    event.notes = 'ISUP 2. Nuevo riesgo: localmente avanzado.'
    assert validate_reclassifications(case, orch.kb) == ([], [])


def test_compact_tnm_can_update_nodes_as_well_as_t_stage(sample):
    orch, case, event = sample
    case.progression.trajectory[1].notes = 'Estadio clínico cT3acN1cM0.'
    event.notes = 'ISUP 2.'
    event.state = 'cn1_m0'
    assert validate_reclassifications(case, orch.kb) == ([], [])


def test_yaml_changes_are_respected_not_hardcoded_thresholds(sample):
    orch, _, _ = sample
    knowledge = orch.kb.get('risk_groups')
    knowledge['risk_groups']['favourable_intermediate']['any'][0]['all']['psa_lt'] = 8.0
    knowledge['risk_groups']['unfavourable_intermediate']['any'][0]['all']['psa_gte'] = 8.0
    state = dict(isup=2, psa=9.0, cT='cT1c', cN='cN0', cM='cM0')
    assert classify(state, knowledge) == 'unfavourable_intermediate'


def test_feedback_regenerates_only_progression_and_preserves_candidate(sample):
    orch, good, _ = sample
    bad = good.model_copy(deep=True)
    event = next(e for e in bad.progression.trajectory if e.event_type == 'active_surveillance_reclassification')
    event.notes = event.notes.replace('intermedio favorable', 'intermedio desfavorable')
    before = bad.model_dump()
    result = orch.validator.run(bad)
    assert not result.passed
    orch._validation_feedback = result.errors
    orch.progression_agent.run = Mock(return_value=good.progression.model_copy(deep=True))
    for name in ('patient_agent', 'diagnostic_agent', 'treatment_agent'):
        getattr(orch, name).run = Mock(side_effect=AssertionError('Frozen baseline must not regenerate'))
    fixed = orch._regenerate_blocks(bad, 'test_source.json')
    assert orch.validator.run(fixed).passed
    orch.progression_agent.run.assert_called_once()
    assert orch.progression_agent.run.call_args.kwargs['validation_feedback'] == result.errors
    assert orch.progression_agent.run.call_args.kwargs['previous_progression'] == bad.progression
    assert bad.model_dump() == before
    for field in ('patient', 'diagnosis', 'risk_classification', 'treatment_plan'):
        assert getattr(fixed, field) == getattr(bad, field)
