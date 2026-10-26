from pathlib import Path

from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.deterministic_validator_agent import DeterministicValidatorAgent
from src.knowledge_base import KnowledgeBase
from src.models import ClinicalEvent


from src.config_loader import PROJECT_ROOT
ROOT = PROJECT_ROOT


def test_planned_docetaxel_cycles_do_not_mean_completed_cycles():
    case = build_case("metastatic_de_novo_triplet_progression_to_mcrpc")
    case.progression.trajectory = [
        ClinicalEvent(month=1, event_type="androgen_deprivation_therapy", state="metastatic_hormone_sensitive",
                      treatment="adt_plus_docetaxel_plus_arpi"),
        ClinicalEvent(month=2, event_type="docetaxel", state="metastatic_hormone_sensitive",
                      treatment="docetaxel", notes="Inicio ciclo 1; se planifican 6 ciclos."),
        ClinicalEvent(month=6, event_type="mHSPC_response", state="metastatic_hormone_sensitive",
                      treatment="adt_plus_arpi", notes="Respuesta tras completar 6 ciclos de docetaxel."),
    ]
    validator = DeterministicValidatorAgent(KnowledgeBase(ROOT))
    assert validator._validate_docetaxel_cycle_timing(case) == []
    case.progression.trajectory[-1].month = 3
    assert validator._validate_docetaxel_cycle_timing(case)


def test_completion_still_checked_when_plan_is_mentioned_in_same_event():
    case = build_case("metastatic_de_novo_triplet_progression_to_mcrpc")
    case.progression.trajectory = [
        ClinicalEvent(month=2, event_type="docetaxel", state="metastatic_hormone_sensitive", notes="Inicio ciclo 1."),
        ClinicalEvent(month=3, event_type="follow_up", state="metastatic_hormone_sensitive",
                      notes="Docetaxel: 6 ciclos planificados, ya completados."),
    ]
    assert DeterministicValidatorAgent(KnowledgeBase(ROOT))._validate_docetaxel_cycle_timing(case)


def build_case(archetype: str):
    orch = BatchOrchestratorAgent(ROOT, use_llm_generation=False, start_index=2300)
    return orch._build_case("test_case", archetype)


def staging_events(risk_group: str):
    result = "cN1_cM0" if risk_group == "cn1_m0" else "cN0_cM0"
    return [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_localised"),
        ClinicalEvent(
            month=0,
            event_type="systemic_staging_before_treatment",
            state=risk_group,
            systemic_staging_modality="psma_pet_ct",
            systemic_staging_result=result,
        ),
        ClinicalEvent(month=1, event_type="multidisciplinary_committee_decision", state=risk_group),
    ]


def test_active_surveillance_in_isup5_rejected():
    case = build_case("low_risk_active_surveillance_stable")
    case.diagnosis.biopsy.isup = 5
    case.diagnosis.biopsy.gleason = "4+5"
    case.risk_classification.risk_group = "low_risk"
    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)
    assert not result.passed
    assert any("no_active_surveillance_for_isup_4_or_5" in e for e in result.errors)


def test_salvage_without_recurrence_rejected():
    case = build_case("favourable_intermediate_radical_prostatectomy_stable")
    case.progression.trajectory.append(
        ClinicalEvent(month=12, event_type="salvage_radiotherapy", state="salvage", treatment="salvage_radiotherapy")
    )
    case.progression.trajectory = sorted(case.progression.trajectory, key=lambda e: e.month)
    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)
    assert not result.passed
    assert any("no_salvage_rt_without_psa_failure_after_rp" in e for e in result.errors)


def test_mcrpc_without_adt_rejected():
    case = build_case("favourable_intermediate_radical_prostatectomy_stable")
    case.progression.trajectory.append(ClinicalEvent(month=12, event_type="mCRPC", state="metastatic_crpc"))
    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)
    assert not result.passed
    assert any("no_mcrpc_without_adt_and_crpc_progression" in e for e in result.errors)


def test_progression_rules_reject_active_surveillance_event_after_prostatectomy():
    case = build_case("favourable_intermediate_radical_prostatectomy_stable")
    case.progression.trajectory.append(
        ClinicalEvent(
            month=12,
            event_type="active_surveillance_follow_up",
            state="post_radical_prostatectomy",
            treatment=None,
            notes="Seguimiento clinico tras prostatectomia radical.",
        )
    )
    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)
    assert not result.passed
    assert any("progression_rule_allowed_previous_states: active_surveillance_follow_up" in e for e in result.errors)


def test_progression_rules_accept_active_surveillance_follow_up_after_active_surveillance():
    case = build_case("low_risk_active_surveillance_stable")
    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)
    assert result.passed


def test_progression_rules_accept_prostatectomy_treatment_marker_before_pathology():
    case = build_case("low_risk_active_surveillance_stable")
    case.archetype = "low_risk_active_surveillance_progression_to_treatment"
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_localised"),
        ClinicalEvent(month=6, event_type="active_surveillance_follow_up", state="active_surveillance"),
        ClinicalEvent(month=18, event_type="active_surveillance_reclassification", state="active_surveillance"),
        ClinicalEvent(
            month=20,
            event_type="switch_to_curative_treatment",
            state="treated_with_curative_intent",
            treatment="radical_prostatectomy",
        ),
        ClinicalEvent(month=21, event_type="postoperative_pathology", state="post_radical_prostatectomy"),
        ClinicalEvent(month=24, event_type="psa_undetectable_after_rp", state="post_radical_prostatectomy"),
    ]
    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)
    assert result.passed


def test_archetype_required_events_are_executable_rules():
    case = build_case("high_risk_rp_positive_margin_salvage_rt")
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_localised"),
        ClinicalEvent(month=2, event_type="radical_prostatectomy", state="treated_with_curative_intent"),
        ClinicalEvent(month=3, event_type="postoperative_pathology", state="post_radical_prostatectomy"),
        ClinicalEvent(month=12, event_type="biochemical_recurrence_after_rp", state="biochemical_recurrence"),
        ClinicalEvent(month=14, event_type="salvage_radiotherapy", state="post_radiotherapy"),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert not result.passed
    assert any("archetype_required_events" in error and "positive_surgical_margin" in error for error in result.errors)


def test_docetaxel_before_mcrpc_allowed_in_mhspc_triplet():
    case = build_case("metastatic_de_novo_triplet_progression_to_mcrpc")
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_metastatic_hormone_sensitive"),
        ClinicalEvent(
            month=0,
            event_type="androgen_deprivation_therapy",
            state="diagnosed_metastatic_hormone_sensitive",
            treatment="adt_plus_docetaxel_plus_arpi",
        ),
        ClinicalEvent(
            month=1,
            event_type="docetaxel",
            state="diagnosed_metastatic_hormone_sensitive",
            treatment="docetaxel",
        ),
        ClinicalEvent(
            month=1,
            event_type="arpi",
            state="diagnosed_metastatic_hormone_sensitive",
            treatment="arpi",
        ),
        ClinicalEvent(month=6, event_type="mHSPC_response", state="diagnosed_metastatic_hormone_sensitive"),
        ClinicalEvent(month=18, event_type="castration_resistant_progression", state="metastatic_crpc"),
        ClinicalEvent(month=19, event_type="mCRPC", state="metastatic_crpc"),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert result.passed


def test_adt_short_context_allowed_for_unfavourable_intermediate():
    case = build_case("unfavourable_intermediate_rt_biochemical_recurrence")

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert result.passed


def test_adt_short_context_rejected_for_high_risk_localised():
    case = build_case("high_risk_rt_long_adt_response")
    case.treatment_plan.initial_treatment = "external_beam_radiotherapy_plus_short_adt"
    case.treatment_plan.treatment_components = ["external_beam_radiotherapy", "short_adt"]
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_localised"),
        ClinicalEvent(
            month=2,
            event_type="external_beam_radiotherapy",
            state="post_radiotherapy",
            treatment="external_beam_radiotherapy_plus_short_adt",
        ),
        ClinicalEvent(
            month=2,
            event_type="androgen_deprivation_therapy",
            state="post_radiotherapy",
            treatment="adt",
        ),
        ClinicalEvent(month=12, event_type="psa_response", state="post_radiotherapy", psa=0.4),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert not result.passed
    assert any("progression_rule_possible_treatments_by_risk_group: androgen_deprivation_therapy" in error for error in result.errors)


def test_arpi_abiraterone_allowed_for_m0_systemic_intensification():
    case = build_case("locally_advanced_multimodal_treatment")
    case.treatment_plan.initial_treatment = "radiotherapy_plus_adt_plus_systemic_intensification"
    case.treatment_plan.treatment_components = ["external_beam_radiotherapy", "adt", "abiraterone"]
    case.progression.trajectory = staging_events("locally_advanced") + [
        ClinicalEvent(
            month=2,
            event_type="local_or_multimodal_treatment",
            state="post_radiotherapy",
            treatment="radiotherapy_plus_adt_plus_systemic_intensification",
        ),
        ClinicalEvent(month=2, event_type="androgen_deprivation_therapy", state="post_radiotherapy", treatment="adt"),
        ClinicalEvent(month=2, event_type="arpi", state="post_radiotherapy", treatment="abiraterone"),
        ClinicalEvent(month=12, event_type="follow_up", state="post_radiotherapy", psa=0.5),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert result.passed


def test_arpi_enzalutamide_rejected_for_m0_systemic_intensification():
    case = build_case("locally_advanced_multimodal_treatment")
    case.treatment_plan.initial_treatment = "radiotherapy_plus_adt_plus_systemic_intensification"
    case.treatment_plan.treatment_components = ["external_beam_radiotherapy", "adt", "enzalutamide"]
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_locally_advanced"),
        ClinicalEvent(
            month=1,
            event_type="local_or_multimodal_treatment",
            state="post_radiotherapy",
            treatment="radiotherapy_plus_adt_plus_systemic_intensification",
        ),
        ClinicalEvent(month=1, event_type="androgen_deprivation_therapy", state="post_radiotherapy", treatment="adt"),
        ClinicalEvent(month=1, event_type="arpi", state="post_radiotherapy", treatment="enzalutamide"),
        ClinicalEvent(month=12, event_type="follow_up", state="post_radiotherapy", psa=0.5),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert not result.passed
    assert any("progression_rule_possible_event_treatments_by_risk_group: arpi" in error for error in result.errors)


def test_arpi_event_rejects_aggregated_treatment_code():
    case = build_case("metastatic_de_novo_adt_arpi_response")
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_metastatic_hormone_sensitive"),
        ClinicalEvent(
            month=0,
            event_type="androgen_deprivation_therapy",
            state="diagnosed_metastatic_hormone_sensitive",
            treatment="adt",
        ),
        ClinicalEvent(
            month=0,
            event_type="arpi",
            state="diagnosed_metastatic_hormone_sensitive",
            treatment="adt_plus_arpi",
        ),
        ClinicalEvent(month=6, event_type="mHSPC_response", state="metastatic_hormone_sensitive", treatment="adt_plus_arpi"),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert not result.passed
    assert any("progression_rule_possible_event_treatments_by_risk_group: arpi" in error for error in result.errors)


def test_cabazitaxel_before_mcrpc_rejected():
    case = build_case("mcrpc_docetaxel_then_cabazitaxel")
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_metastatic_hormone_sensitive"),
        ClinicalEvent(
            month=0,
            event_type="androgen_deprivation_therapy",
            state="diagnosed_metastatic_hormone_sensitive",
            treatment="adt_plus_docetaxel",
        ),
        ClinicalEvent(month=6, event_type="docetaxel", state="diagnosed_metastatic_hormone_sensitive"),
        ClinicalEvent(month=12, event_type="cabazitaxel", state="diagnosed_metastatic_hormone_sensitive"),
        ClinicalEvent(month=18, event_type="castration_resistant_progression", state="metastatic_crpc"),
        ClinicalEvent(month=19, event_type="mCRPC", state="metastatic_crpc"),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert not result.passed
    assert any("no_mcrpc_treatment_before_mcrpc" in error for error in result.errors)


def test_docetaxel_cabazitaxel_rules_use_current_state_not_initial_risk_group():
    case = build_case("mcrpc_docetaxel_then_cabazitaxel")
    case.risk_classification.risk_group = "high_risk_localised"
    case.treatment_plan.initial_treatment = "external_beam_radiotherapy_plus_long_adt"
    case.treatment_plan.treatment_components = ["external_beam_radiotherapy", "long_adt"]
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_localised"),
        ClinicalEvent(
            month=6,
            event_type="androgen_deprivation_therapy",
            state="metastatic_hormone_sensitive",
            treatment="external_beam_radiotherapy_plus_long_adt",
        ),
        ClinicalEvent(month=18, event_type="castration_resistant_progression", state="metastatic_crpc"),
        ClinicalEvent(month=19, event_type="mCRPC", state="metastatic_crpc"),
        ClinicalEvent(month=20, event_type="docetaxel", state="metastatic_crpc", treatment="docetaxel"),
        ClinicalEvent(month=26, event_type="cabazitaxel", state="metastatic_crpc", treatment="cabazitaxel"),
    ]

    errors = DeterministicValidatorAgent(KnowledgeBase(ROOT))._validate_progression_rules(case)

    assert errors == []


def test_cabazitaxel_rejects_non_mcrpc_current_state():
    case = build_case("mcrpc_docetaxel_then_cabazitaxel")
    case.progression.trajectory = [
        ClinicalEvent(month=0, event_type="diagnosis", state="diagnosed_metastatic_hormone_sensitive"),
        ClinicalEvent(
            month=0,
            event_type="androgen_deprivation_therapy",
            state="diagnosed_metastatic_hormone_sensitive",
            treatment="adt_plus_docetaxel",
        ),
        ClinicalEvent(month=18, event_type="castration_resistant_progression", state="metastatic_crpc"),
        ClinicalEvent(month=19, event_type="mCRPC", state="metastatic_crpc"),
        ClinicalEvent(month=20, event_type="docetaxel", state="metastatic_crpc", treatment="docetaxel"),
        ClinicalEvent(month=26, event_type="cabazitaxel", state="metastatic_hormone_sensitive", treatment="cabazitaxel"),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert not result.passed
    assert any("progression_rule_allowed_current_states: cabazitaxel" in error for error in result.errors)


def test_radical_prostatectomy_rejects_plnd_mentions():
    case = build_case("unfavourable_intermediate_rp_biochemical_recurrence")
    case.treatment_plan.initial_treatment = "radical_prostatectomy"
    case.treatment_plan.treatment_components = ["radical_prostatectomy"]
    case.progression.trajectory = staging_events("high_risk_localised") + [
        ClinicalEvent(
            month=2,
            event_type="radical_prostatectomy",
            state="post_radical_prostatectomy",
            treatment="radical_prostatectomy",
            notes="Prostatectomia radical con linfadenectomia pelvica.",
        ),
        ClinicalEvent(month=3, event_type="postoperative_pathology", state="post_radical_prostatectomy"),
        ClinicalEvent(month=4, event_type="psa_undetectable_after_rp", state="post_radical_prostatectomy", psa=0.01),
        ClinicalEvent(month=24, event_type="biochemical_recurrence_after_rp", state="biochemical_recurrence", psa=0.4),
        ClinicalEvent(month=27, event_type="salvage_radiotherapy", state="post_radiotherapy", treatment="salvage_radiotherapy"),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert not result.passed
    assert any("prostatectomy_plnd_consistency" in error for error in result.errors)


def test_radical_prostatectomy_plus_plnd_accepts_consistent_plnd_coding():
    case = build_case("high_risk_rp_positive_margin_salvage_rt")
    case.treatment_plan.initial_treatment = "radical_prostatectomy_plus_plnd"
    case.treatment_plan.treatment_components = ["radical_prostatectomy", "pelvic_lymph_node_dissection"]
    case.progression.trajectory = staging_events("high_risk_localised") + [
        ClinicalEvent(
            month=2,
            event_type="radical_prostatectomy",
            state="post_radical_prostatectomy",
            treatment="radical_prostatectomy_plus_plnd",
            notes="Prostatectomia radical con linfadenectomia pelvica.",
        ),
        ClinicalEvent(month=3, event_type="postoperative_pathology", state="post_radical_prostatectomy"),
        ClinicalEvent(month=4, event_type="positive_surgical_margin", state="post_radical_prostatectomy"),
        ClinicalEvent(month=5, event_type="psa_undetectable_after_rp", state="post_radical_prostatectomy", psa=0.01),
        ClinicalEvent(month=18, event_type="biochemical_recurrence_after_rp", state="biochemical_recurrence", psa=0.4),
        ClinicalEvent(month=20, event_type="salvage_radiotherapy", state="post_radiotherapy", treatment="salvage_radiotherapy"),
    ]

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert result.passed


def test_biopsy_pathology_fields_missing_emit_warning_not_error():
    case = build_case("low_risk_active_surveillance_stable")
    case.diagnosis.biopsy.tumour_extent = None
    case.diagnosis.biopsy.lymphovascular_invasion_if_present = None
    case.diagnosis.biopsy.extraprostatic_extension_if_present = None

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert result.passed
    assert any("biopsy_pathology_required_fields_missing" in warning for warning in result.warnings)


def test_biopsy_pathology_fields_present_do_not_emit_warning():
    case = build_case("low_risk_active_surveillance_stable")
    case.diagnosis.biopsy.tumour_extent = "low_volume"
    case.diagnosis.biopsy.lymphovascular_invasion_if_present = "not_reported"
    case.diagnosis.biopsy.extraprostatic_extension_if_present = "not_identified"

    result = DeterministicValidatorAgent(KnowledgeBase(ROOT)).run(case)

    assert result.passed
    assert not any("biopsy_pathology_required_fields_missing" in warning for warning in result.warnings)
