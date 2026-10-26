import json
from unittest.mock import Mock

import pytest

from src.base_agent import BaseAgent
from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.deterministic_feedback import repair_scope
from src.export_agent import ExportAgent
from src.batch_orchestrator_agent import ArchetypeGateError


def sample(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch._artifact_prefix = "test_"
    archetype = "low_risk_active_surveillance_stable"
    orch.dataset_plan["quotas_by_archetype"] = {archetype: 1}
    orch.dataset_plan["max_attempts_per_case"] = 2
    good = orch._build_case("caso_sintetico_2300", archetype)
    assert orch.validator.run(good).passed
    return orch, good


@pytest.mark.parametrize("error,scope", [
    ("patient_schema: age: invalid", "patient"),
    ("psa_density_consistency: mismatch", "diagnosis"),
    ("gleason_isup_consistency: mismatch", "diagnosis"),
    ("treatment_allowed_by_risk_group: mismatch", "treatment"),
    ("no_low_risk_if_psa_above_20", "risk"),
    ("events_sorted_by_month", "progression"),
    ("no_parp_without_biomarker", None),
    ("prostatectomy_plnd_consistency: ambiguous", None),
    ("unknown_rule: not registered", None),
])
def test_explicit_ownership(error, scope):
    assert repair_scope([error]) == scope


def test_density_correction_only_calls_diagnostic_and_full_validation(tmp_path):
    orch, good = sample(tmp_path)
    bad = good.model_copy(deep=True)
    bad.diagnosis.psa_density = 99
    before = bad.model_dump()
    orch._build_case = Mock(return_value=bad)
    orch.diagnostic_agent.run = Mock(return_value=good.diagnosis.model_copy(deep=True))
    for name in ("patient_agent", "treatment_agent", "progression_agent"):
        getattr(orch, name).run = Mock(side_effect=AssertionError("unnecessary call"))
    validate = Mock(wraps=orch.validator.run)
    orch.validator.run = validate
    accepted = orch.run(1)
    assert len(accepted) == 1
    assert validate.call_count == 2
    orch._build_case.assert_called_once()
    orch.diagnostic_agent.run.assert_called_once()
    call = orch.diagnostic_agent.run.call_args
    assert call.kwargs["validation_feedback"] == ["psa_density_consistency: PSA density no coincide aproximadamente con PSA/volumen"]
    assert call.kwargs["previous_diagnosis"].psa_density == 99
    assert bad.model_dump() == before
    assert accepted[0].patient == bad.patient
    assert accepted[0].treatment_plan == bad.treatment_plan
    assert accepted[0].progression == bad.progression


def test_treatment_correction_keeps_patient_diagnosis_and_refreshes_progression(tmp_path):
    orch, good = sample(tmp_path)
    bad = good.model_copy(deep=True)
    bad.treatment_plan.initial_treatment = "radical_prostatectomy"
    errors = orch.validator.run(bad).errors
    assert orch._repair_scope(bad, errors) == "treatment"
    orch._validation_feedback = errors
    orch.patient_agent.run = Mock(side_effect=AssertionError("patient frozen"))
    orch.diagnostic_agent.run = Mock(side_effect=AssertionError("diagnosis frozen"))
    orch.treatment_agent.run = Mock(return_value=good.treatment_plan)
    orch.progression_agent.run = Mock(return_value=good.progression)
    fixed = orch._regenerate_blocks(bad, "source.json")
    assert fixed.patient == bad.patient and fixed.diagnosis == bad.diagnosis
    assert orch.validator.run(fixed).passed
    orch.treatment_agent.run.assert_called_once()
    orch.progression_agent.run.assert_called_once()


def test_patient_schema_feedback_cannot_change_valid_biomarkers(tmp_path):
    orch, good = sample(tmp_path)
    bad = good.model_copy(deep=True)
    bad.patient.age = 101
    result = orch.validator.run(bad)
    assert not result.passed and result.errors[0].startswith("patient_schema: age:")
    orch._validation_feedback = result.errors
    changed = good.patient.model_copy(deep=True)
    changed.brca2_status = "positive" if good.patient.brca2_status != "positive" else "negative"
    orch.patient_agent.run = Mock(return_value=changed)
    with pytest.raises(ValueError, match="valid fields or biomarkers"):
        orch._regenerate_blocks(bad, "source.json")
    assert bad.patient.age == 101


def test_density_fix_cannot_change_other_diagnostic_fields(tmp_path):
    orch, good = sample(tmp_path)
    bad = good.model_copy(deep=True)
    bad.diagnosis.psa_density = 99
    orch._validation_feedback = orch.validator.run(bad).errors
    wrong = good.diagnosis.model_copy(deep=True)
    wrong.initial_psa += 1
    orch.diagnostic_agent.run = Mock(return_value=wrong)
    with pytest.raises(ValueError, match="unrelated diagnostic"):
        orch._regenerate_blocks(bad, "source.json")


def test_feedback_prompt_only_appended_on_correction():
    agent = BaseAgent("test", None)
    assert agent.correction_prompt("ORIGINAL", None, None) == "ORIGINAL"
    assert "exact_rule: evidence" in agent.correction_prompt("ORIGINAL", ["exact_rule: evidence"], None)


def test_invalid_diagnosis_schema_is_reported_without_arithmetic_exception(tmp_path):
    orch, good = sample(tmp_path)
    good.diagnosis.prostate_volume_cc = 0
    result = orch.validator.run(good)
    assert not result.passed
    assert any(error.startswith("diagnosis_schema: prostate_volume_cc:") for error in result.errors)


def test_patient_feedback_rebuilds_dependent_blocks_not_other_cases(tmp_path):
    orch, good = sample(tmp_path)
    bad = good.model_copy(deep=True)
    bad.patient.age = 101
    before = bad.model_dump()
    orch._validation_feedback = orch.validator.run(bad).errors
    orch.patient_agent.run = Mock(return_value=good.patient.model_copy(deep=True))
    orch.diagnostic_agent.run = Mock(return_value=good.diagnosis.model_copy(deep=True))
    orch.treatment_agent.run = Mock(return_value=good.treatment_plan.model_copy(deep=True))
    orch.progression_agent.run = Mock(return_value=good.progression.model_copy(deep=True))
    fixed = orch._regenerate_blocks(bad, "source.json")
    assert orch.validator.run(fixed).passed
    assert fixed.case_id == bad.case_id and fixed.archetype == bad.archetype
    assert bad.model_dump() == before
    for name in ("patient_agent", "diagnostic_agent", "treatment_agent", "progression_agent"):
        getattr(orch, name).run.assert_called_once()
    assert "patient_schema: age:" in orch.patient_agent.run.call_args.kwargs["validation_feedback"][0]


def test_mixed_diagnosis_progression_feedback_goes_to_each_owner(tmp_path):
    orch, good = sample(tmp_path)
    bad = good.model_copy(deep=True)
    bad.diagnosis.psa_density = 99
    bad.progression.trajectory.reverse()
    orch._validation_feedback = orch.validator.run(bad).errors
    assert orch._repair_scope(bad, orch._validation_feedback) == "diagnosis"
    orch.diagnostic_agent.run = Mock(return_value=good.diagnosis)
    orch.progression_agent.run = Mock(return_value=good.progression)
    orch.treatment_agent.run = Mock(side_effect=AssertionError("treatment still valid"))
    fixed = orch._regenerate_blocks(bad, "source.json")
    assert orch.validator.run(fixed).passed
    assert all(e.startswith("psa_density_consistency:") for e in orch.diagnostic_agent.run.call_args.kwargs["validation_feedback"])
    assert not any(e.startswith("psa_density_consistency:") for e in orch.progression_agent.run.call_args.kwargs["validation_feedback"])


def test_early_treatment_failure_retries_only_treatment_then_progression(tmp_path):
    orch, good = sample(tmp_path)
    wrong = good.treatment_plan.model_copy(deep=True)
    wrong.initial_treatment = "radical_prostatectomy"
    orch.patient_agent.run = Mock(return_value=good.patient)
    orch.diagnostic_agent.run = Mock(return_value=good.diagnosis)
    orch.treatment_agent.run = Mock(side_effect=[wrong, good.treatment_plan])
    orch.progression_agent.run = Mock(return_value=good.progression)
    result = orch.run(1)
    assert len(result) == 1
    orch.patient_agent.run.assert_called_once()
    orch.diagnostic_agent.run.assert_called_once()
    assert orch.treatment_agent.run.call_count == 2
    orch.progression_agent.run.assert_called_once()
    assert "archetype_required_treatment" in " ".join(orch.treatment_agent.run.call_args.kwargs["validation_feedback"])


def test_corrected_treatment_is_not_repeated_if_progression_call_fails(tmp_path):
    orch, good = sample(tmp_path)
    wrong = good.treatment_plan.model_copy(deep=True)
    wrong.initial_treatment = "radical_prostatectomy"
    orch.treatment_agent.run = Mock(side_effect=[wrong, good.treatment_plan])
    with pytest.raises(ArchetypeGateError) as caught:
        orch._build_case(good.case_id, good.archetype)
    orch._treatment_retry = caught.value.baseline
    orch._validation_feedback = caught.value.errors
    orch.progression_agent.run = Mock(side_effect=[ValueError("temporary progression error"), good.progression])
    with pytest.raises(ValueError, match="temporary progression"):
        orch._build_case(good.case_id, good.archetype)
    assert orch._treatment_retry["treatment_verified"]
    orch._build_case(good.case_id, good.archetype)
    assert orch.treatment_agent.run.call_count == 2
    assert orch.progression_agent.run.call_count == 2
