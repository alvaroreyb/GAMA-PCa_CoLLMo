import json
from unittest.mock import Mock

import pytest

from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.export_agent import ExportAgent
from src.models import KimiReviewResult, ValidationResult


def setup_case(tmp_path):
    orch = BatchOrchestratorAgent(use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch.dataset_plan["max_attempts_per_case"] = 2
    archetype = "high_risk_rp_positive_margin_salvage_rt"
    orch.dataset_plan["quotas_by_archetype"] = {archetype: 1}
    good = orch._build_case("caso_sintetico_2300", archetype)
    assert orch.validator.run(good).passed
    bad = good.model_copy(deep=True)
    bad.progression.trajectory = [e for e in bad.progression.trajectory if e.event_type != "positive_surgical_margin"]
    assert not orch.validator.run(bad).passed
    return orch, good, bad


def test_only_progression_is_called_again_and_whole_case_is_validated(tmp_path, monkeypatch):
    orch, good, bad = setup_case(tmp_path)
    original = bad.model_dump()
    build = Mock(return_value=bad)
    progression = Mock(return_value=good.progression.model_copy(deep=True))
    validate = Mock(wraps=orch.validator.run)
    monkeypatch.setattr(orch, "_build_case", build)
    monkeypatch.setattr(orch.progression_agent, "run", progression)
    monkeypatch.setattr(orch.validator, "run", validate)
    for name in ("patient_agent", "diagnostic_agent", "treatment_agent"):
        monkeypatch.setattr(getattr(orch, name), "run", Mock(side_effect=AssertionError("paid baseline regeneration")))
    result = orch.run(1)
    assert len(result) == 1
    build.assert_called_once()
    progression.assert_called_once()
    assert validate.call_count == 2
    assert bad.model_dump() == original
    assert result[0].model_dump(exclude={"progression"}) == bad.model_dump(exclude={"progression"})
    assert "positive_surgical_margin" in " ".join(progression.call_args.kwargs["validation_feedback"])
    assert progression.call_args.kwargs["previous_progression"].model_dump() == bad.progression.model_dump()
    scopes = list((tmp_path / "generated_artifacts").glob("*00_retry_scope.json"))
    assert json.loads(scopes[0].read_text())["scope"] == "progression_only"


@pytest.mark.parametrize("error", [
    "unknown_future_rule: error",
    "no_parp_without_biomarker", "prostatectomy_plnd_consistency: mixed blocks",
    "progression_rule_cM_required: event inconsistent with diagnosis",
    "progression_rule_possible_treatments_by_risk_group: ambiguous initial treatment",
])
def test_mixed_or_ambiguous_errors_regenerate_everything(tmp_path, monkeypatch, error):
    orch, _, bad = setup_case(tmp_path)
    build = Mock(return_value=bad)
    progression = Mock(side_effect=AssertionError("not progression-only"))
    monkeypatch.setattr(orch, "_build_case", build)
    monkeypatch.setattr(orch.progression_agent, "run", progression)
    monkeypatch.setattr(orch.validator, "run", Mock(return_value=ValidationResult(
        passed=False, errors=["archetype_required_events: missing margin", error])))
    assert orch.run(1) == []
    assert build.call_count == 2
    progression.assert_not_called()


def test_repaired_progression_can_still_be_rejected(tmp_path, monkeypatch):
    orch, _, bad = setup_case(tmp_path)
    monkeypatch.setattr(orch, "_build_case", Mock(return_value=bad))
    monkeypatch.setattr(orch.progression_agent, "run", Mock(return_value=bad.progression.model_copy(deep=True)))
    assert orch.run(1) == []
    report = json.loads(orch.batch_report_path.read_text())
    assert report["accepted"] == 0 and report["rejected"] == 1


def test_resume_reuses_last_invalid_progression_without_overwriting_artifact(tmp_path, monkeypatch):
    orch, good, bad = setup_case(tmp_path)
    orch.dataset_plan["max_attempts_per_case"] = 1
    original = tmp_path / "generated_artifacts/caso_sintetico_2300_attempt_01_01_structured_case.json"
    original.write_text(json.dumps({"case": bad.model_dump()}))
    content = original.read_bytes()
    (tmp_path / "batch_2300_2300.json").write_text(json.dumps({
        "status": "paused_pilot_limit", "planned": 1,
        "processed": 0, "accepted": 0, "rejected": 0, "outcomes": [],
    }))
    build = Mock(side_effect=AssertionError("baseline must be restored"))
    monkeypatch.setattr(orch, "_build_case", build)
    monkeypatch.setattr(orch.progression_agent, "run", Mock(return_value=good.progression.model_copy(deep=True)))
    assert len(orch.run(1, resume=True)) == 1
    build.assert_not_called()
    assert original.read_bytes() == content


def test_changed_baseline_is_not_eligible_for_retry(tmp_path):
    orch, _, bad = setup_case(tmp_path)
    bad.risk_classification.risk_group = "low_risk"
    validation = ValidationResult(passed=False, errors=["archetype_required_events: missing margin"])
    assert not orch._can_retry_progression(bad, validation)


def test_agent_cannot_mutate_frozen_baseline(tmp_path, monkeypatch):
    orch, good, bad = setup_case(tmp_path)
    before = bad.model_dump()
    def mutate(patient, *args, **kwargs):
        patient.age += 1
        return good.progression
    monkeypatch.setattr(orch.progression_agent, "run", mutate)
    orch._artifact_prefix = "test_"
    orch._validation_feedback = []
    with pytest.raises(ValueError, match="frozen baseline"):
        orch._regenerate_progression(bad, "test")
    assert bad.model_dump() == before


def test_kimi_rejection_does_not_trigger_progression_only_retry(tmp_path, monkeypatch):
    orch, good, _ = setup_case(tmp_path)
    build = Mock(return_value=good)
    progression = Mock(side_effect=AssertionError("ambiguous clinical-review scope"))
    monkeypatch.setattr(orch, "_build_case", build)
    monkeypatch.setattr(orch.progression_agent, "run", progression)
    monkeypatch.setattr(orch, "_run_kimi", Mock(return_value=KimiReviewResult(
        passed=False, errors=["clinical-review failure"])))
    assert orch.run(1) == []
    assert build.call_count == 2
    progression.assert_not_called()


def test_resume_uses_newest_candidate_for_diagnosis_and_progression_feedback(tmp_path):
    orch, _, bad = setup_case(tmp_path)
    older = tmp_path / "generated_artifacts/caso_sintetico_2300_attempt_01_01_structured_case.json"
    older.write_text(json.dumps({"case": bad.model_dump()}))
    newest = tmp_path / "generated_artifacts/caso_sintetico_2300_attempt_02_01_structured_case.json"
    mixed = bad.model_copy(deep=True)
    mixed.diagnosis.psa_density = 99
    newest.write_text(json.dumps({"case": mixed.model_dump()}))
    import os
    os.utime(older, ns=(1, 1))
    os.utime(newest, ns=(2, 2))
    loaded = orch._load_progression_retry(bad.case_id, bad.archetype)
    assert loaded[0].diagnosis.psa_density == 99
    assert loaded[2] == newest.name
    assert orch._repair_scope(loaded[0], loaded[1]) == "diagnosis"


def test_progression_repair_exception_retries_without_rebuilding_baseline(tmp_path, monkeypatch):
    orch, good, bad = setup_case(tmp_path)
    orch.dataset_plan["max_attempts_per_case"] = 3
    build = Mock(return_value=bad)
    progression = Mock(side_effect=[ValueError("invalid progression response"), good.progression])
    monkeypatch.setattr(orch, "_build_case", build)
    monkeypatch.setattr(orch.progression_agent, "run", progression)
    assert len(orch.run(1)) == 1
    assert build.call_count == 1
    assert progression.call_count == 2


def test_resume_candidate_now_valid_skips_all_generation_but_runs_review(tmp_path, monkeypatch):
    orch, good, _ = setup_case(tmp_path)
    case_path = tmp_path / "generated_artifacts/caso_sintetico_2300_attempt_01_01_structured_case.json"
    case_path.write_text(json.dumps({"case": good.model_dump()}))
    validation_path = case_path.with_name(case_path.name.replace("01_structured_case", "02_deterministic_validation"))
    validation_path.write_text(json.dumps({"validation": {"passed": False, "errors": ["old_false_positive"]}}))
    (tmp_path / "batch_2300_2300.json").write_text(json.dumps({
        "status": "paused_repeated_validation_errors", "planned": 1,
        "processed": 0, "accepted": 0, "rejected": 0, "outcomes": [],
    }))
    build = Mock(side_effect=AssertionError("no generation required"))
    progression = Mock(side_effect=AssertionError("no progression correction required"))
    review = Mock(return_value=KimiReviewResult(passed=True))
    monkeypatch.setattr(orch, "_build_case", build)
    monkeypatch.setattr(orch.progression_agent, "run", progression)
    monkeypatch.setattr(orch, "_run_kimi", review)
    assert len(orch.run(1, resume=True)) == 1
    build.assert_not_called()
    progression.assert_not_called()
    review.assert_called_once()


def test_resume_valid_downstream_rejection_is_not_reused(tmp_path):
    orch, good, _ = setup_case(tmp_path)
    case_path = tmp_path / "generated_artifacts/caso_sintetico_2300_attempt_01_01_structured_case.json"
    case_path.write_text(json.dumps({"case": good.model_dump()}))
    validation_path = case_path.with_name(case_path.name.replace("01_structured_case", "02_deterministic_validation"))
    validation_path.write_text(json.dumps({"validation": {"passed": True, "errors": []}}))
    case_path.with_name(case_path.name.replace("01_structured_case", "03_kimi_case_review")).write_text(
        json.dumps({"review": {"passed": False, "errors": ["clinical contradiction"]}}))
    assert orch._load_progression_retry(good.case_id, good.archetype) is None
