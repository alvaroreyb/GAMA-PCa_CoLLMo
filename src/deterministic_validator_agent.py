from __future__ import annotations

from typing import Any
import re

from .models import SyntheticCase, ValidationResult, PatientProfile, Diagnosis, TreatmentPlan, ProgressionPlan
from pydantic import ValidationError


CURATIVE_EVENTS = {"radical_prostatectomy", "external_beam_radiotherapy", "external_beam_radiotherapy_plus_short_adt", "external_beam_radiotherapy_plus_long_adt"}
POST_MCRPC_ONLY_TREATMENTS = {"cabazitaxel", "parp_inhibitor", "lutetium_psma"}

# Only rules whose failing data can be changed in progression. Cross-block
# rules (biomarkers, cM requirements, PLND, initial-treatment compatibility)
# intentionally fall back to full generation. Unknown/new rules do too.
PROGRESSION_ONLY_RULES = frozenset({
    "no_salvage_rt_without_psa_failure_after_rp", "no_bcr_before_curative_treatment",
    "no_mcrpc_without_adt_and_crpc_progression", "no_mcrpc_treatment_before_mcrpc",
    "events_sorted_by_month", "archetype_required_events", "archetype_forbidden_events",
    "progression_event_vocabulary", "docetaxel_cycle_timing", "phoenix_nadir_consistency",
    "systemic_staging_before_committee_or_treatment", "systemic_staging_requires_previous_diagnosis",
    "systemic_staging_requires_accepted_modality", "systemic_staging_requires_documented_result",
    "cn1_m0_requires_documented_cn1_m0_staging_result",
    "localised_m0_groups_require_negative_distant_staging_result",
    "progression_rule_required_previous_events", "progression_rule_required_previous_events_any",
    "progression_rule_allowed_previous_events", "progression_rule_allowed_previous_events_any",
    "progression_rule_allowed_previous_states", "progression_rule_required_previous_states",
    "progression_rule_allowed_current_states", "progression_rule_allowed_risk_groups",
    "progression_rule_possible_event_treatments_by_risk_group",
    "progression_rule_required_previous_treatments_any",
    "reclassification_risk_consistency",
})


class DeterministicValidatorAgent:
    name = "deterministic_validator"

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def validate_archetype_baseline(self, archetype: str, risk: str, treatment: str) -> list[str]:
        """Same rules for the early gate and the final full validation."""
        errors = []
        rules = self.kb.archetype(archetype)
        if risk not in rules.get("target_risk_groups", []):
            errors.append(f"archetype_risk_group_compatibility: {archetype} no permite {risk}")
        required = rules.get("required_treatment")
        if required and treatment != required:
            errors.append(f"archetype_required_treatment: {archetype} requiere {required}; actual={treatment}")
        allowed = set(rules.get("allowed_treatments", []) or [])
        if allowed and treatment not in allowed:
            errors.append(f"archetype_allowed_treatments: {archetype} no permite {treatment}; "
                          f"permitidos={', '.join(sorted(allowed))}")
        return errors

    @staticmethod
    def progression_only_failure(result: ValidationResult) -> bool:
        """A full validation found no failing baseline or ambiguous rule."""
        return (not result.passed and bool(result.errors)
                and all(error.split(":", 1)[0] in PROGRESSION_ONLY_RULES for error in result.errors))

    def run(self, case: SyntheticCase) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        # Recheck the existing schemas: models can be mutated after parsing.
        # This adds no clinical criterion and avoids arithmetic on invalid data.
        for block, model in (("patient", PatientProfile), ("diagnosis", Diagnosis),
                             ("treatment_plan", TreatmentPlan), ("progression", ProgressionPlan)):
            try:
                model.model_validate(getattr(case, block).model_dump())
            except ValidationError as exc:
                for issue in exc.errors():
                    location = ".".join(str(part) for part in issue["loc"])
                    errors.append(f"{block}_schema: {location}: {issue['msg']}")
        if errors:
            return ValidationResult(passed=False, errors=errors)
        d = case.diagnosis
        risk = case.risk_classification.risk_group
        treatment = case.treatment_plan.initial_treatment
        events = case.progression.trajectory
        event_types = [e.event_type for e in events]
        event_treatments = [e.treatment for e in events if e.treatment]
        all_treatments = [treatment] + event_treatments

        mapping = self.kb.gleason_to_isup()
        gleason = d.biopsy.gleason
        isup = d.biopsy.isup
        if gleason and isup and mapping.get(gleason) != isup:
            errors.append(f"gleason_isup_consistency: {gleason} no corresponde a ISUP {isup}")
        warnings.extend(self._validate_biopsy_pathology_fields(case))

        expected_density = d.initial_psa / d.prostate_volume_cc
        if abs(d.psa_density - expected_density) > max(0.02, expected_density * 0.15):
            errors.append("psa_density_consistency: PSA density no coincide aproximadamente con PSA/volumen")

        if risk == "low_risk" and d.initial_psa > 20:
            errors.append("no_low_risk_if_psa_above_20")
        if risk == "low_risk" and (d.tnm.cT.startswith("cT3") or d.tnm.cT.startswith("cT4")):
            errors.append("no_low_risk_if_ct3_or_ct4")
        if treatment.startswith("active_surveillance") and isup in {4, 5}:
            errors.append("no_active_surveillance_for_isup_4_or_5")
        if treatment.startswith("active_surveillance") and d.tnm.cM.startswith("cM1"):
            errors.append("no_active_surveillance_for_metastatic_disease")

        if "salvage_radiotherapy" in event_types:
            salvage_idx = event_types.index("salvage_radiotherapy")
            previous = set(event_types[:salvage_idx])
            if not previous.intersection({"persistent_psa_after_rp", "biochemical_recurrence_after_rp"}):
                errors.append("no_salvage_rt_without_psa_failure_after_rp")

        for idx, ev in enumerate(event_types):
            if ev in {"biochemical_recurrence_after_rp", "biochemical_recurrence_after_rt"}:
                if not set(event_types[:idx]).intersection(CURATIVE_EVENTS):
                    errors.append("no_bcr_before_curative_treatment")

        if "mCRPC" in event_types:
            m_idx = event_types.index("mCRPC")
            previous_types = set(event_types[:m_idx])
            previous_treatments = set(t for t in all_treatments[:]) | previous_types
            has_adt = bool(previous_types.intersection({"androgen_deprivation_therapy"}) or any("adt" in str(t).lower() for t in previous_treatments))
            if not has_adt or "castration_resistant_progression" not in previous_types:
                errors.append("no_mcrpc_without_adt_and_crpc_progression")

        if any(t == "parp_inhibitor" for t in all_treatments):
            notes = " ".join(e.notes or "" for e in events).lower()
            if case.patient.brca2_status != "positive" and "hrr" not in notes:
                errors.append("no_parp_without_biomarker")

        if d.tnm.cM == "cM0" and d.metastatic_sites != "none" and d.metastatic_sites != "pelvic_nodes":
            errors.append("no_cm0_with_metastatic_sites")
        if d.tnm.cM.startswith("cM1") and d.metastatic_sites == "none":
            errors.append("no_cm1_without_metastatic_sites")

        staging_errors, staging_warnings = self._validate_systemic_staging(case)
        errors.extend(staging_errors)
        warnings.extend(staging_warnings)
        errors.extend(self._validate_docetaxel_cycle_timing(case))
        errors.extend(self._validate_phoenix_nadir_consistency(case))
        from .reclassification_validation import validate_reclassifications
        reclassification_errors, reclassification_warnings = validate_reclassifications(case, self.kb)
        errors.extend(reclassification_errors)
        warnings.extend(reclassification_warnings)

        months = [e.month for e in events]
        if months != sorted(months):
            errors.append("events_sorted_by_month")

        if treatment not in self.kb.allowed_treatments(risk):
            errors.append(f"treatment_allowed_by_risk_group: {treatment} no permitido para {risk}")

        errors.extend(self._validate_prostatectomy_plnd_consistency(case))

        errors.extend(self.validate_archetype_baseline(case.archetype, risk, treatment))
        archetype_rules = self.kb.archetype(case.archetype)

        required_events = set(archetype_rules.get("required_events", []) or [])
        seen_event_markers = self._case_event_markers(case)
        missing_required_events = sorted(required_events - seen_event_markers)
        if missing_required_events:
            errors.append(
                f"archetype_required_events: {case.archetype} requiere eventos "
                f"{', '.join(missing_required_events)}"
            )

        forbidden_events = set(archetype_rules.get("forbidden_events", []))
        forbidden_seen = sorted(forbidden_events.intersection(event_types))
        if forbidden_seen:
            errors.append(f"archetype_forbidden_events: {case.archetype} no permite {', '.join(forbidden_seen)}")

        errors.extend(self._validate_progression_rules(case))
        errors.extend(self._validate_event_vocabulary(case))

        seen_mcrpc = False
        for e in events:
            if e.event_type == "mCRPC":
                seen_mcrpc = True
            if (
                e.event_type in POST_MCRPC_ONLY_TREATMENTS
                or e.treatment in POST_MCRPC_ONLY_TREATMENTS
            ) and not seen_mcrpc:
                errors.append("no_mcrpc_treatment_before_mcrpc")

        return ValidationResult(passed=not errors, errors=errors, warnings=warnings)


    def _validate_systemic_staging(self, case: SyntheticCase) -> tuple[list[str], list[str]]:
        config_by_risk = self.kb.get("diagnostic_pathways").get("systemic_staging_by_risk_group", {}) or {}
        risk = case.risk_classification.risk_group
        config = config_by_risk.get(risk, {}) or {}
        severity = config.get("missing_documentation_severity", "none")
        events = case.progression.trajectory
        event_types = [event.event_type for event in events]
        staging_indices = [idx for idx, event in enumerate(events) if event.event_type == "systemic_staging_before_treatment"]
        errors: list[str] = []
        warnings: list[str] = []

        trigger_events = {
            "multidisciplinary_committee_decision",
            "radical_prostatectomy",
            "external_beam_radiotherapy",
            "local_or_multimodal_treatment",
            "local_treatment",
            "androgen_deprivation_therapy",
            "systemic_intensification",
        }
        first_trigger_idx = next((idx for idx, event in enumerate(events) if event.event_type in trigger_events), None)
        has_staging_before_trigger = bool(
            staging_indices
            and (
                first_trigger_idx is None
                or any(idx < first_trigger_idx for idx in staging_indices)
            )
        )

        if severity in {"error", "warning"} and first_trigger_idx is not None and not has_staging_before_trigger:
            msg = (
                "systemic_staging_before_committee_or_treatment: "
                f"{risk} requiere evento systemic_staging_before_treatment antes de comité o tratamiento inicial"
            )
            if severity == "error":
                errors.append(msg)
            else:
                warnings.append(msg)

        if risk in {"low_risk", "favourable_intermediate"} and staging_indices:
            presentation = case.patient.presentation_context.lower()
            if "bone_pain" not in presentation and "dolor" not in presentation and "advanced" not in presentation:
                warnings.append(
                    "no_routine_systemic_staging_for_low_or_favourable_intermediate: "
                    f"{risk} no debería generar estadificación sistémica rutinaria salvo sospecha clínica"
                )

        definitions = self.kb.get("progression_rules").get("event_definitions", {}) or {}
        staging_rule = definitions.get("systemic_staging_before_treatment", {}) or {}
        allowed_modalities = set(staging_rule.get("allowed_modalities", []) or [])
        allowed_results = set(staging_rule.get("allowed_results", []) or [])

        for event in (events[idx] for idx in staging_indices):
            modality = event.systemic_staging_modality
            result = event.systemic_staging_result
            if allowed_modalities and modality not in allowed_modalities:
                errors.append(
                    "systemic_staging_requires_accepted_modality: "
                    f"modalidad={modality}; permitidas={', '.join(sorted(allowed_modalities))}"
                )
            if allowed_results and result not in allowed_results:
                errors.append(
                    "systemic_staging_requires_documented_result: "
                    f"resultado={result}; permitidos={', '.join(sorted(allowed_results))}"
                )

            if risk == "cn1_m0" and result != "cN1_cM0":
                errors.append(
                    "cn1_m0_requires_documented_cn1_m0_staging_result: "
                    f"resultado={result}; esperado=cN1_cM0"
                )
            if risk in {"unfavourable_intermediate", "high_risk_localised", "locally_advanced"} and result != "cN0_cM0":
                errors.append(
                    "localised_m0_groups_require_negative_distant_staging_result: "
                    f"resultado={result}; esperado=cN0_cM0 para mantener {risk}"
                )

        if "systemic_staging_before_treatment" in event_types:
            first_staging_idx = event_types.index("systemic_staging_before_treatment")
            if "diagnosis" not in event_types[:first_staging_idx]:
                errors.append("systemic_staging_requires_previous_diagnosis")

        return errors, warnings

    def _validate_docetaxel_cycle_timing(self, case: SyntheticCase) -> list[str]:
        events = case.progression.trajectory
        errors: list[str] = []
        for idx, event in enumerate(events):
            text = " ".join(part for part in (event.treatment, event.notes) if part).lower()
            if "docetaxel" not in text:
                continue
            match = re.search(r"(\d+)\s*ciclos", text)
            if not match:
                continue
            # A planned course is not a report that all cycles have been delivered.
            planned = re.search(r"planific|previst|programad|se plan|se prev", text)
            completed = re.search(r"completad|tras completar|tras finalizar|finalizaci[oó]n|tras terminar|recibid|administrad", text)
            if planned and not completed:
                continue
            cycles = int(match.group(1))
            if cycles < 2:
                continue
            # Mentions of a combination regimen in ADT/committee events are not cycle 1.
            previous_docetaxel_months = [prev.month for prev in events[:idx] if prev.event_type == "docetaxel"]
            if not previous_docetaxel_months:
                continue
            elapsed_months = event.month - min(previous_docetaxel_months)
            minimum_months = max(1, round(((cycles - 1) * 21) / 30.44))
            if elapsed_months < minimum_months:
                errors.append(
                    "docetaxel_cycle_timing: "
                    f"{cycles} ciclos q3w no caben entre mes {min(previous_docetaxel_months)} y mes {event.month}"
                )
        return errors

    def _validate_phoenix_nadir_consistency(self, case: SyntheticCase) -> list[str]:
        errors: list[str] = []
        events = case.progression.trajectory
        for idx, event in enumerate(events):
            if event.event_type != "biochemical_recurrence_after_rt":
                continue
            notes = (event.notes or "").lower()
            if "phoenix" not in notes or "nadir" not in notes:
                continue
            match = re.search(r"nadir[^0-9]*(\d+(?:[\.,]\d+)?)", notes)
            if not match:
                continue
            stated_nadir = float(match.group(1).replace(",", "."))
            prior_psa = [prev.psa for prev in events[:idx] if prev.psa is not None]
            if not prior_psa:
                continue
            actual_nadir = min(prior_psa)
            if stated_nadir > actual_nadir + 0.05:
                errors.append(
                    "phoenix_nadir_consistency: "
                    f"nadir citado={stated_nadir}; nadir mínimo previo documentado={actual_nadir}"
                )
        return errors

    def _validate_biopsy_pathology_fields(self, case: SyntheticCase) -> list[str]:
        biopsy = case.diagnosis.biopsy
        if not biopsy.performed:
            return []

        required_fields = self.kb.get("biopsy_rules").get("pathology_required_fields", []) or []
        missing: list[str] = []
        for field_name in required_fields:
            if not hasattr(biopsy, field_name):
                missing.append(field_name)
                continue
            value = getattr(biopsy, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field_name)

        if not missing:
            return []

        return [
            "biopsy_pathology_required_fields_missing: "
            "biopsy_rules.yaml:pathology_required_fields sin valor en "
            f"{', '.join(missing)}"
        ]

    def _validate_progression_rules(self, case: SyntheticCase) -> list[str]:
        rules = self.kb.get("progression_rules")
        definitions = rules.get("event_definitions", {}) or {}
        known_states = set(rules.get("states", []) or [])
        events = case.progression.trajectory
        event_types = [e.event_type for e in events]
        errors: list[str] = []

        for idx, event in enumerate(events):
            rule = definitions.get(event.event_type)
            if not rule:
                continue

            previous_event_set = self._previous_event_markers(case, idx)
            previous_states = [e.state for e in events[:idx]]
            previous_state = previous_states[-1] if previous_states else None
            current_state = event.state
            previous_known_states = {state for state in previous_states if state in known_states}
            previous_treatments = self._previous_treatments(case, idx)

            errors.extend(self._validate_required_previous_events(event.event_type, rule, previous_event_set))
            errors.extend(self._validate_allowed_previous_events(event.event_type, rule, previous_event_set))
            errors.extend(
                self._validate_previous_states(
                    event.event_type,
                    rule,
                    previous_state,
                    current_state,
                    previous_known_states,
                    known_states,
                )
            )
            errors.extend(self._validate_current_state(event.event_type, rule, current_state))
            errors.extend(self._validate_allowed_risk_group(event.event_type, rule, case.risk_classification.risk_group))
            errors.extend(
                self._validate_possible_treatments_by_risk_group(
                    event.event_type,
                    rule,
                    case.risk_classification.risk_group,
                    case,
                    event.treatment,
                )
            )
            errors.extend(
                self._validate_possible_event_treatments_by_risk_group(
                    event.event_type,
                    rule,
                    case.risk_classification.risk_group,
                    event.treatment,
                )
            )
            errors.extend(self._validate_required_previous_treatments(event.event_type, rule, previous_treatments))
            errors.extend(self._validate_cm_requirement(event.event_type, rule, case.diagnosis.tnm.cM))
            errors.extend(self._validate_required_biomarker(event.event_type, rule, case))

        return errors

    def _validate_required_previous_events(
        self,
        event_type: str,
        rule: dict[str, Any],
        previous_event_set: set[str],
    ) -> list[str]:
        errors: list[str] = []
        required = set(rule.get("required_previous_events", []) or [])
        missing = sorted(required - previous_event_set)
        if missing:
            errors.append(
                f"progression_rule_required_previous_events: {event_type} requiere eventos previos {', '.join(missing)}"
            )

        required_any = set(rule.get("required_previous_events_any", []) or [])
        if required_any and not previous_event_set.intersection(required_any):
            errors.append(
                f"progression_rule_required_previous_events_any: {event_type} requiere uno de {', '.join(sorted(required_any))}"
            )
        return errors

    def _validate_allowed_previous_events(
        self,
        event_type: str,
        rule: dict[str, Any],
        previous_event_set: set[str],
    ) -> list[str]:
        errors: list[str] = []
        allowed = set(rule.get("allowed_previous_events", []) or [])
        if allowed and not previous_event_set.intersection(allowed):
            errors.append(
                f"progression_rule_allowed_previous_events: {event_type} requiere antecedente en {', '.join(sorted(allowed))}"
            )

        allowed_any = set(rule.get("allowed_previous_events_any", []) or [])
        if allowed_any and not previous_event_set.intersection(allowed_any):
            errors.append(
                f"progression_rule_allowed_previous_events_any: {event_type} requiere antecedente en {', '.join(sorted(allowed_any))}"
            )
        return errors

    def _validate_previous_states(
        self,
        event_type: str,
        rule: dict[str, Any],
        previous_state: str | None,
        current_state: str | None,
        previous_known_states: set[str],
        known_states: set[str],
    ) -> list[str]:
        errors: list[str] = []
        allowed_states = set(rule.get("allowed_previous_states", []) or [])
        if allowed_states and current_state in allowed_states:
            return errors

        state_to_check = previous_state if previous_state in known_states else current_state
        if allowed_states and state_to_check in known_states and state_to_check not in allowed_states:
            errors.append(
                f"progression_rule_allowed_previous_states: {event_type} requiere estado previo en "
                f"{', '.join(sorted(allowed_states))}; previo={state_to_check}"
            )

        required_states = set(rule.get("required_previous_states", []) or [])
        if required_states and previous_known_states and not previous_known_states.intersection(required_states):
            errors.append(
                f"progression_rule_required_previous_states: {event_type} requiere estado previo en "
                f"{', '.join(sorted(required_states))}"
            )
        return errors

    def _validate_allowed_risk_group(self, event_type: str, rule: dict[str, Any], risk_group: str) -> list[str]:
        allowed = set(rule.get("allowed_risk_groups", []) or [])
        if allowed and risk_group not in allowed:
            return [
                f"progression_rule_allowed_risk_groups: {event_type} no permitido para {risk_group}; "
                f"permitidos={', '.join(sorted(allowed))}"
            ]
        return []

    def _validate_current_state(self, event_type: str, rule: dict[str, Any], current_state: str | None) -> list[str]:
        allowed = set(rule.get("allowed_current_states", []) or [])
        if allowed and current_state not in allowed:
            return [
                f"progression_rule_allowed_current_states: {event_type} requiere estado actual en "
                f"{', '.join(sorted(allowed))}; actual={current_state}"
            ]
        return []

    def _validate_possible_treatments_by_risk_group(
        self,
        event_type: str,
        rule: dict[str, Any],
        risk_group: str,
        case: SyntheticCase,
        event_treatment: str | None,
    ) -> list[str]:
        treatments_by_risk = rule.get("possible_treatments_by_risk_group", {}) or {}
        if not treatments_by_risk:
            return []

        allowed = set(treatments_by_risk.get(risk_group, []) or [])
        if not allowed:
            return [
                f"progression_rule_possible_treatments_by_risk_group: {event_type} no define tratamientos para {risk_group}"
            ]

        candidates = {
            case.treatment_plan.initial_treatment,
            *(case.treatment_plan.treatment_components or []),
        }
        if event_treatment:
            candidates.add(event_treatment)

        if candidates.intersection(allowed):
            return []

        return [
            f"progression_rule_possible_treatments_by_risk_group: {event_type} no compatible con {risk_group}; "
            f"tratamientos={', '.join(sorted(candidates))}; permitidos={', '.join(sorted(allowed))}"
        ]

    def _validate_possible_event_treatments_by_risk_group(
        self,
        event_type: str,
        rule: dict[str, Any],
        risk_group: str,
        event_treatment: str | None,
    ) -> list[str]:
        treatments_by_risk = rule.get("possible_event_treatments_by_risk_group", {}) or {}
        if not treatments_by_risk or not event_treatment:
            return []

        allowed = set(treatments_by_risk.get(risk_group, []) or [])
        if not allowed:
            return [
                f"progression_rule_possible_event_treatments_by_risk_group: {event_type} no define tratamientos de evento para {risk_group}"
            ]

        if event_treatment in allowed:
            return []

        return [
            f"progression_rule_possible_event_treatments_by_risk_group: {event_type} treatment={event_treatment} "
            f"no compatible con {risk_group}; permitidos={', '.join(sorted(allowed))}"
        ]

    def _validate_required_previous_treatments(
        self,
        event_type: str,
        rule: dict[str, Any],
        previous_treatments: set[str],
    ) -> list[str]:
        required_any = set(rule.get("required_previous_treatments_any", []) or [])
        if required_any and not self._has_any_treatment(previous_treatments, required_any):
            return [
                f"progression_rule_required_previous_treatments_any: {event_type} requiere tratamiento previo en "
                f"{', '.join(sorted(required_any))}"
            ]
        return []

    def _validate_cm_requirement(self, event_type: str, rule: dict[str, Any], cM: str) -> list[str]:
        required = rule.get("cM_required")
        if required and cM != required:
            return [f"progression_rule_cM_required: {event_type} requiere {required}; actual={cM}"]

        required_any = set(rule.get("cM_required_any", []) or [])
        if required_any and cM not in required_any:
            return [f"progression_rule_cM_required_any: {event_type} requiere uno de {', '.join(sorted(required_any))}; actual={cM}"]
        return []

    def _validate_required_biomarker(self, event_type: str, rule: dict[str, Any], case: SyntheticCase) -> list[str]:
        required_any = set(rule.get("required_biomarker_any", []) or [])
        if not required_any:
            return []

        available: set[str] = set()
        if case.patient.brca2_status == "positive":
            available.add("brca2_positive")
        notes = " ".join(e.notes or "" for e in case.progression.trajectory).lower()
        if "hrr" in notes:
            available.add("hrr_alteration_positive")

        if not available.intersection(required_any):
            return [
                f"progression_rule_required_biomarker_any: {event_type} requiere biomarcador en "
                f"{', '.join(sorted(required_any))}"
            ]
        return []

    def _previous_treatments(self, case: SyntheticCase, event_index: int) -> set[str]:
        values = [case.treatment_plan.initial_treatment]
        values.extend(e.treatment for e in case.progression.trajectory[:event_index] if e.treatment)
        values.extend(e.event_type for e in case.progression.trajectory[:event_index])
        return {self._norm(value) for value in values if value}

    def _previous_event_markers(self, case: SyntheticCase, event_index: int) -> set[str]:
        values: list[str | None] = []
        for event in case.progression.trajectory[:event_index]:
            values.append(event.event_type)
            values.append(event.treatment)
        return {value for value in values if value}

    def _case_event_markers(self, case: SyntheticCase) -> set[str]:
        values: list[str | None] = [case.treatment_plan.initial_treatment]
        values.extend(case.treatment_plan.treatment_components)
        for event in case.progression.trajectory:
            values.append(event.event_type)
            values.append(event.treatment)
        return {value for value in values if value}

    def _validate_event_vocabulary(self, case: SyntheticCase) -> list[str]:
        allowed = set(self.kb.get("progression_rules").get("event_definitions", {}) or {})
        allowed.update(self.kb.archetype(case.archetype).get("required_events", []) or [])
        for archetype_rules in self.kb.archetypes.values():
            allowed.update(archetype_rules.get("required_events", []) or [])

        follow_up_rules = self.kb.get("follow_up_rules")
        for context in (follow_up_rules.get("follow_up_contexts", {}) or {}).values():
            allowed.update(context.get("possible_outcomes", []) or [])

        unknown = sorted({event.event_type for event in case.progression.trajectory if event.event_type not in allowed})
        if unknown:
            return [f"progression_event_vocabulary: eventos no definidos en clinical_knowledge: {', '.join(unknown)}"]
        return []

    def _validate_prostatectomy_plnd_consistency(self, case: SyntheticCase) -> list[str]:
        treatment = case.treatment_plan.initial_treatment
        if treatment not in {"radical_prostatectomy", "radical_prostatectomy_plus_plnd"}:
            return []

        errors: list[str] = []
        components = set(case.treatment_plan.treatment_components)
        event_treatments = {event.treatment for event in case.progression.trajectory if event.treatment}
        text = " ".join(
            value
            for value in [
                case.patient.presentation_context,
                case.treatment_plan.rationale,
                *(event.notes or "" for event in case.progression.trajectory),
            ]
            if value
        ).lower()
        mentions_plnd = any(
            marker in text
            for marker in (
                "linfadenectomia",
                "linfadenectomía",
                "plnd",
                "pelvic lymph node dissection",
                "diseccion ganglionar pelvica",
                "disección ganglionar pélvica",
            )
        )
        component_has_plnd = bool(components.intersection({"pelvic_lymph_node_dissection", "plnd"}))
        event_has_plus_plnd = "radical_prostatectomy_plus_plnd" in event_treatments

        if treatment == "radical_prostatectomy":
            if mentions_plnd or component_has_plnd or event_has_plus_plnd:
                errors.append(
                    "prostatectomy_plnd_consistency: radical_prostatectomy no debe mezclar linfadenectomia/PLND; "
                    "usar radical_prostatectomy_plus_plnd o retirar PLND del caso"
                )
            return errors

        if treatment == "radical_prostatectomy_plus_plnd":
            if "radical_prostatectomy" in event_treatments:
                errors.append(
                    "prostatectomy_plnd_consistency: radical_prostatectomy_plus_plnd no debe codificar el evento quirurgico "
                    "como radical_prostatectomy simple"
                )
            if not component_has_plnd and not event_has_plus_plnd and not mentions_plnd:
                errors.append(
                    "prostatectomy_plnd_consistency: radical_prostatectomy_plus_plnd debe reflejar PLND en componentes, "
                    "evento quirurgico o notas"
                )
        return errors

    def _has_any_treatment(self, previous_treatments: set[str], required_any: set[str]) -> bool:
        normalized_required = {self._norm(value) for value in required_any}
        for required in normalized_required:
            if required in previous_treatments:
                return True
            if any(required in treatment or treatment in required for treatment in previous_treatments):
                return True
        return False

    @staticmethod
    def _norm(value: str) -> str:
        return str(value).strip().lower()
