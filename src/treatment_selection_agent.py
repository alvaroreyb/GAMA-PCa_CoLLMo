from __future__ import annotations

from .base_agent import BaseAgent, compact_json
from .models import Diagnosis, PatientProfile, RiskClassification, TreatmentPlan, ValidationResult


class TreatmentValidationError(ValueError):
    def __init__(self, plan, risk):
        super().__init__(f"Treatment {plan.initial_treatment} not allowed for {risk}")
        self.plan = plan
        self.errors = [f"treatment_allowed_by_risk_group: {plan.initial_treatment} no permitido para {risk}"]


class TreatmentSelectionAgent(BaseAgent):
    def run(self, patient: PatientProfile, diagnosis: Diagnosis, risk: RiskClassification, archetype: str, knowledge_base, use_llm: bool = True,
            validation_feedback=None, previous_treatment: TreatmentPlan | None = None) -> TreatmentPlan:
        if not use_llm or not self.ollama_client:
            plan = self._fallback(archetype, risk.risk_group, knowledge_base)
        else:
            prompt = self.prompt_template().format(
                patient=compact_json(patient.model_dump()),
                diagnosis=compact_json(diagnosis.model_dump()),
                risk=compact_json(risk.model_dump()),
                archetype=archetype,
                treatment_options=compact_json(knowledge_base.get("treatment_options")),
                archetype_rules=compact_json(knowledge_base.archetype(archetype)),
            )
            prompt = self.correction_prompt(prompt, validation_feedback, previous_treatment)
            raw = self.ollama_client.chat(self.model_name, self.build_messages(prompt), self.temperature, self.num_predict, json_format=True, call_name=self.name)
            plan = self.parse_json_response(raw, TreatmentPlan, TreatmentPlan.model_json_schema())
        if plan.initial_treatment not in knowledge_base.allowed_treatments(risk.risk_group):
            raise TreatmentValidationError(plan, risk.risk_group)
        return plan

    @staticmethod
    def _fallback(archetype: str, risk_group: str, kb) -> TreatmentPlan:
        rules = kb.archetype(archetype)
        treatment = rules.get("required_treatment") or (rules.get("allowed_treatments") or [None])[0]
        if not treatment:
            if risk_group == "metastatic_hormone_sensitive_at_diagnosis" and "docetaxel" in archetype:
                treatment = "adt_plus_docetaxel"
            elif risk_group == "metastatic_hormone_sensitive_at_diagnosis":
                treatment = "adt_plus_arpi"
            else:
                treatment = next(iter(kb.allowed_treatments(risk_group)))
        intent = "surveillance" if "surveillance" in treatment or treatment.startswith("watchful") else "disease_control" if "adt" in treatment else "curative"
        if "palliative" in archetype:
            intent = "palliative"
        return TreatmentPlan(
            initial_treatment=treatment,
            treatment_intent=intent,
            treatment_components=treatment.split("_plus_"),
            rationale=f"Tratamiento compatible con el arquetipo {archetype} y el grupo {risk_group}.",
        )
