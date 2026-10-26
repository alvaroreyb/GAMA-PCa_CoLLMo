"""Explicit ownership of deterministic errors; no clinical-review routing."""
from .deterministic_validator_agent import PROGRESSION_ONLY_RULES

DIAGNOSIS_RULES = frozenset({
    "diagnosis_schema", "gleason_isup_consistency", "psa_density_consistency",
    "no_cm0_with_metastatic_sites", "no_cm1_without_metastatic_sites",
    "archetype_risk_group_compatibility",
})
TREATMENT_RULES = frozenset({
    "treatment_plan_schema", "treatment_allowed_by_risk_group",
    "archetype_required_treatment", "archetype_allowed_treatments",
    "no_active_surveillance_for_isup_4_or_5", "no_active_surveillance_for_metastatic_disease",
})
RISK_RULES = frozenset({"no_low_risk_if_psa_above_20", "no_low_risk_if_ct3_or_ct4"})
ORDER = ("patient", "diagnosis", "risk", "treatment", "progression")


def feedback_by_owner(errors):
    grouped = {name: [] for name in ORDER}
    for error in errors:
        rule = error.split(":", 1)[0]
        if rule == "patient_schema":
            owner = "patient"
        elif rule in DIAGNOSIS_RULES:
            owner = "diagnosis"
        elif rule in RISK_RULES:
            owner = "risk"
        elif rule in TREATMENT_RULES:
            owner = "treatment"
        elif rule in PROGRESSION_ONLY_RULES or rule == "progression_schema":
            owner = "progression"
        else:
            return None  # Biomarkers, PLND, unknown/cross-block causes: no guessing.
        grouped[owner].append(error)
    return grouped


def repair_scope(errors):
    grouped = feedback_by_owner(errors)
    if grouped is None:
        return None
    return next((owner for owner in ORDER if grouped[owner]), None)
