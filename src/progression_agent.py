from __future__ import annotations

import random

from .base_agent import BaseAgent, compact_json
from .models import ClinicalEvent, Diagnosis, FinalStatus, PatientProfile, ProgressionPlan, RiskClassification, TreatmentPlan


class ProgressionAgent(BaseAgent):
    def run(
        self,
        patient: PatientProfile,
        diagnosis: Diagnosis,
        risk: RiskClassification,
        treatment: TreatmentPlan,
        archetype: str,
        knowledge_base,
        use_llm: bool = True,
        rng: random.Random | None = None,
        validation_feedback: list[str] | None = None,
        previous_progression: ProgressionPlan | None = None,
    ) -> ProgressionPlan:
        if not use_llm or not self.ollama_client:
            return self._fallback(diagnosis, risk, treatment, archetype)
        prompt = self.prompt_template().format(
            patient=compact_json(patient.model_dump()),
            diagnosis=compact_json(diagnosis.model_dump()),
            risk=compact_json(risk.model_dump()),
            treatment=compact_json(treatment.model_dump()),
            archetype=archetype,
            archetype_rules=compact_json(knowledge_base.archetype(archetype)),
            progression_rules=compact_json(knowledge_base.get("progression_rules")),
            follow_up_rules=compact_json(knowledge_base.get("follow_up_rules")),
        )
        if 'active_surveillance_reclassification' in knowledge_base.archetype(archetype).get('required_events', []):
            prompt += (
                '\nREGLAS YAML PARA CLASIFICAR EL RIESGO TRAS RECLASIFICACION:\n'
                + compact_json(knowledge_base.get('risk_groups'))
            )
        if validation_feedback:
            prompt += "\nERRORES DEL CANDIDATO ANTERIOR A CORREGIR:\n" + "\n".join(validation_feedback)
        if previous_progression is not None:
            prompt += (
                "\nCORRECCIÓN LOCALIZADA DEL MISMO CASO:\n"
                "Paciente, diagnóstico, riesgo y plan de tratamiento son bloques fijos. "
                "Corrige exclusivamente la progresión adjunta y devuelve el bloque ProgressionPlan completo. "
                "Conserva los eventos y valores válidos; modifica solo lo necesario para cumplir las reglas. "
                "No inventes cambios basales ni alteraciones moleculares para sortear un error.\n"
                "PROGRESIÓN ANTERIOR:\n" + compact_json(previous_progression.model_dump())
            )
        raw = self.ollama_client.chat(self.model_name, self.build_messages(prompt), self.temperature, self.num_predict, json_format=True, call_name=self.name)
        plan = self.parse_json_response(raw, ProgressionPlan, ProgressionPlan.model_json_schema())
        return self._normalize_postoperative_timing(plan)

    def _fallback(self, diagnosis: Diagnosis, risk: RiskClassification, treatment: TreatmentPlan, archetype: str) -> ProgressionPlan:
        events = [
            ClinicalEvent(month=0, event_type="diagnosis", state="diagnosticado", psa=diagnosis.initial_psa, imaging="RM multiparametrica prostatica", treatment=None, complication=None, notes="Diagnostico inicial."),
        ]
        t = treatment.initial_treatment
        if risk.risk_group in {"unfavourable_intermediate", "high_risk_localised", "locally_advanced", "cn1_m0"}:
            staging_result = "cN1_cM0" if risk.risk_group == "cn1_m0" else "cN0_cM0"
            events.append(
                ClinicalEvent(
                    month=0,
                    event_type="systemic_staging_before_treatment",
                    state=risk.risk_group,
                    psa=diagnosis.initial_psa,
                    imaging="PET/TC-PSMA sin metastasis a distancia" if staging_result == "cN0_cM0" else "PET/TC-PSMA con afectacion ganglionar pelvica regional y sin metastasis a distancia",
                    treatment=None,
                    complication=None,
                    notes="Estudio de estadificacion sistemica previo a decision terapeutica.",
                    systemic_staging_modality="psma_pet_ct",
                    systemic_staging_result=staging_result,
                )
            )
            events.append(
                ClinicalEvent(
                    month=1,
                    event_type="multidisciplinary_committee_decision",
                    state=risk.risk_group,
                    psa=diagnosis.initial_psa,
                    imaging=None,
                    treatment=None,
                    complication=None,
                    notes="Decision en comite uro-oncologico tras completar estadificacion sistemica.",
                )
            )
        if "focal_hifu" in t:
            events += [
                ClinicalEvent(month=1, event_type="shared_decision_focal_therapy", state="diagnosed_localised", treatment=None, notes="Decision compartida tras explicar alternativas estandar y caracter no convencional de HIFU focal."),
                ClinicalEvent(month=1, event_type="prospective_registry_enrolment", state="diagnosed_localised", treatment=None, notes="Inclusion en registro prospectivo de terapia focal."),
                ClinicalEvent(month=2, event_type="focal_hifu", state="post_focal_therapy", treatment="focal_hifu", notes="HIFU focal sobre lesion indice visible."),
                ClinicalEvent(month=8, event_type="post_focal_psa_follow_up", state="post_focal_therapy", psa=max(round(diagnosis.initial_psa * 0.45, 1), 0.8), notes="Seguimiento con PSA tras HIFU focal."),
                ClinicalEvent(month=18, event_type="consecutive_psa_rise", state="post_focal_therapy", psa=max(round(diagnosis.initial_psa * 0.75, 1), 1.2), notes="Ascensos consecutivos de PSA en seguimiento."),
                ClinicalEvent(month=20, event_type="post_focal_mp_mri", state="post_focal_therapy", imaging="RMmp con sospecha de recurrencia local limitada", notes="RMmp indicada por ascensos consecutivos de PSA."),
                ClinicalEvent(month=21, event_type="biopsy_confirmed_local_recurrence", state="post_focal_therapy", notes="Biopsia dirigida confirma recurrencia local limitada."),
                ClinicalEvent(month=23, event_type="repeat_focal_hifu", state="post_focal_retreatment", treatment="repeat_focal_hifu", notes="Segundo HIFU focal por recurrencia local limitada."),
                ClinicalEvent(month=30, event_type="post_retreatment_follow_up", state="post_focal_retreatment", psa=max(round(diagnosis.initial_psa * 0.35, 1), 0.8), notes="Seguimiento estable tras retratamiento focal."),
            ]
        elif "active_surveillance" in t:
            events.append(ClinicalEvent(month=6, event_type="active_surveillance_follow_up", state="vigilancia estable", psa=max(round(diagnosis.initial_psa - 0.3, 1), 0.1), testosterone=None, imaging="Sin cambios relevantes en control", treatment="active_surveillance", complication=None, notes="PSA y clinica sin criterios de progresion."))
            if "progression_to_treatment" in archetype:
                events += [
                    ClinicalEvent(month=18, event_type="active_surveillance_reclassification", state="reclasificacion", psa=9.8, testosterone=None, imaging="Nueva lesion PI-RADS 4", treatment=None, complication=None, notes="Aumento de carga tumoral en biopsia de control."),
                    ClinicalEvent(month=20, event_type="switch_to_curative_treatment", state="tratamiento radical indicado", psa=None, testosterone=None, imaging=None, treatment="radical_prostatectomy", complication=None, notes="Cambio a tratamiento con intencion curativa."),
                    ClinicalEvent(month=21, event_type="radical_prostatectomy", state="postoperatorio", psa=None, testosterone=None, imaging=None, treatment="radical_prostatectomy", complication=None, notes=None),
                    ClinicalEvent(month=22, event_type="postoperative_pathology", state="seguimiento postoperatorio", psa=0.03, testosterone=None, imaging=None, treatment=None, complication=None, notes="AP postoperatoria sin datos de enfermedad residual macroscopica. Margenes no informados como positivos."),
                ]
            else:
                events += [
                    ClinicalEvent(month=12, event_type="active_surveillance_follow_up", state="vigilancia estable", psa=round(diagnosis.initial_psa + 0.1, 1), testosterone=None, imaging="RMmp estable, sin nueva lesion sospechosa", treatment="active_surveillance", complication=None, notes="TR sin cambios. Se mantiene pauta de vigilancia."),
                    ClinicalEvent(month=24, event_type="active_surveillance_follow_up", state="reevaluacion protocolizada", psa=round(diagnosis.initial_psa + 0.4, 1), testosterone=None, imaging="RMmp sin progresion local", treatment="active_surveillance", complication=None, notes="Biopsia de reevaluacion sin progresion de grado ni aumento significativo de carga tumoral."),
                    ClinicalEvent(month=36, event_type="active_surveillance_follow_up", state="vigilancia estable", psa=round(diagnosis.initial_psa + 0.6, 1), testosterone=None, imaging=None, treatment="active_surveillance", complication=None, notes="Asintomatico. Sin dolor oseo ni sintomas constitucionales."),
                    ClinicalEvent(month=48, event_type="active_surveillance_follow_up", state="vigilancia estable prolongada", psa=round(diagnosis.initial_psa + 0.8, 1), testosterone=None, imaging="RMmp estable respecto a controles previos", treatment="active_surveillance", complication=None, notes="Continuar seguimiento con PSA, TR y RMmp segun protocolo."),
                ]
        elif "radical_prostatectomy" in t:
            events += [
                ClinicalEvent(month=2, event_type="radical_prostatectomy", state="postoperatorio", psa=None, testosterone=None, imaging=None, treatment=t, complication=None, notes=None),
                ClinicalEvent(month=3, event_type="postoperative_pathology", state="seguimiento postoperatorio", psa=0.04, testosterone=None, imaging=None, treatment=None, complication=None, notes="Anatomia patologica postoperatoria registrada."),
                ClinicalEvent(month=4, event_type="psa_undetectable_after_rp", state="post_radical_prostatectomy", psa=0.01, testosterone=None, imaging=None, treatment=None, complication=None, notes="PSA indetectable tras prostatectomia radical."),
            ]
            if "positive_margin" in archetype:
                events.append(ClinicalEvent(month=3, event_type="positive_surgical_margin", state="factor anatomopatologico adverso", psa=0.04, testosterone=None, imaging=None, treatment=None, complication=None, notes="Margen quirurgico positivo."))
            if "biochemical_recurrence" in archetype or "salvage" in archetype or "positive_margin" in archetype:
                events += [
                    ClinicalEvent(month=18, event_type="biochemical_recurrence_after_rp", state="recaida bioquimica", psa=0.35, testosterone=None, imaging="Sin metastasis en imagen", treatment=None, complication=None, notes=None),
                    ClinicalEvent(month=20, event_type="salvage_radiotherapy", state="tratamiento de rescate", psa=0.31, testosterone=None, imaging=None, treatment="salvage_radiotherapy", complication=None, notes=None),
                ]
        elif "radiotherapy" in t or "local_treatment" in t or t == "radical_prostatectomy_selected":
            event_type = "local_or_multimodal_treatment" if archetype == "locally_advanced_multimodal_treatment" or t in {"radical_prostatectomy_selected", "radiotherapy_plus_adt_plus_systemic_intensification"} else ("local_treatment" if "local_treatment" in t else "external_beam_radiotherapy")
            events.append(ClinicalEvent(month=2, event_type=event_type, state="tratado", psa=diagnosis.initial_psa, testosterone=None, imaging=None, treatment=t, complication=None, notes=None))
            if "adt" in t or "systemic" in t:
                events.append(ClinicalEvent(month=2, event_type="androgen_deprivation_therapy", state="tratamiento sistemico", psa=diagnosis.initial_psa, testosterone=20.0, imaging=None, treatment="adt", complication=None, notes=None))
            if "biochemical_recurrence" in archetype:
                events.append(ClinicalEvent(month=30, event_type="biochemical_recurrence_after_rt", state="recaida bioquimica", psa=4.2, testosterone=280.0, imaging="Sin metastasis", treatment=None, complication=None, notes=None))
            else:
                response_event = "follow_up" if archetype == "locally_advanced_multimodal_treatment" else "psa_response"
                events.append(ClinicalEvent(month=12, event_type=response_event, state="respuesta", psa=0.6, testosterone=20.0 if "adt" in t else None, imaging=None, treatment=None, complication=None, notes="Respuesta bioquimica y seguimiento estable."))
        if diagnosis.tnm.cM.startswith("cM1") or "mcrpc" in archetype:
            if not any(e.event_type == "androgen_deprivation_therapy" for e in events):
                events.append(ClinicalEvent(month=1, event_type="androgen_deprivation_therapy", state="tratamiento sistemico", psa=diagnosis.initial_psa, testosterone=18.0, imaging=None, treatment="adt", complication=None, notes=None))
            events.append(ClinicalEvent(month=4, event_type="mHSPC_response", state="respuesta", psa=3.1, testosterone=16.0, imaging="Respuesta parcial", treatment=t, complication=None, notes=None))
            if "progression_to_mcrpc" in archetype or "mcrpc" in archetype:
                events += [
                    ClinicalEvent(month=24, event_type="castration_resistant_progression", state="progresion resistente a castracion", psa=12.0, testosterone=18.0, imaging="Progresion radiologica", treatment=None, complication=None, notes=None),
                    ClinicalEvent(month=25, event_type="mCRPC", state="CPRC metastasico", psa=14.5, testosterone=18.0, imaging="Enfermedad metastasica resistente a castracion", treatment=None, complication=None, notes=None),
                ]
            if "docetaxel" in archetype:
                events.append(ClinicalEvent(month=26, event_type="docetaxel", state="metastatic_crpc", psa=14.5, testosterone=18.0, imaging=None, treatment="docetaxel", complication=None, notes=None))
            if "cabazitaxel" in archetype:
                events.append(ClinicalEvent(month=34, event_type="cabazitaxel", state="metastatic_crpc", psa=28.0, testosterone=16.0, imaging="Progresion tras docetaxel", treatment="cabazitaxel", complication=None, notes=None))
            if "parp" in archetype:
                events.append(ClinicalEvent(month=26, event_type="parp_inhibitor", state="metastatic_crpc", psa=16.0, testosterone=15.0, imaging=None, treatment="parp_inhibitor", complication=None, notes="Alteracion HRR/BRCA documentada."))
            if "palliative" in archetype:
                events += [
                    ClinicalEvent(month=28, event_type="bone_complication", state="metastasis oseas sintomaticas", psa=34.0, testosterone=17.0, imaging="Metastasis oseas multiples", treatment=None, complication="dolor oseo y evento esqueletico", notes=None),
                    ClinicalEvent(month=30, event_type="palliative_care", state="cuidados paliativos", psa=None, testosterone=None, imaging=None, treatment="palliative_care", complication=None, notes="Control sintomatico prioritario."),
                ]
        events = sorted(events, key=lambda e: e.month)
        status = "mCRPC" if any(e.event_type == "mCRPC" for e in events) else ("stable" if not any("recurrence" in e.event_type for e in events) else "recurrence_treated")
        plan = ProgressionPlan(trajectory=events, final_status=FinalStatus(oncological_status=status, follow_up_status="active_follow_up"))
        plan = self._normalize_systemic_staging_fields(plan, risk)
        return self._normalize_postoperative_timing(plan)

    @staticmethod
    def _normalize_systemic_staging_fields(plan: ProgressionPlan, risk: RiskClassification) -> ProgressionPlan:
        for event in plan.trajectory:
            if event.event_type != "systemic_staging_before_treatment":
                continue
            evidence = " ".join(part for part in (event.imaging, event.notes) if part).lower()
            if event.systemic_staging_modality is None:
                if "psma" in evidence or "pet" in evidence:
                    event.systemic_staging_modality = "psma_pet_ct"
                elif ("tc" in evidence or "ct" in evidence or "tac" in evidence) and "gammagraf" in evidence:
                    event.systemic_staging_modality = "abdominopelvic_ct_plus_bone_scan"
                elif ("rm" in evidence or "mri" in evidence) and "gammagraf" in evidence:
                    event.systemic_staging_modality = "abdominopelvic_mri_plus_bone_scan"
            if event.systemic_staging_result is None:
                compact = evidence.replace("_", " ")
                if risk.risk_group == "cn1_m0" or "cn1 cm0" in compact or "cn1 cM0" in evidence:
                    event.systemic_staging_result = "cN1_cM0"
                elif "cn0 cm0" in compact or "cn0" in compact and "cm0" in compact or "sin metast" in compact or "negativ" in compact:
                    event.systemic_staging_result = "cN0_cM0"
        return plan

    @staticmethod
    def _normalize_postoperative_timing(plan: ProgressionPlan) -> ProgressionPlan:
        events = list(plan.trajectory)
        rp_months = [event.month for event in events if event.event_type == "radical_prostatectomy"]
        if not rp_months:
            return plan

        rp_month = min(rp_months)
        min_pathology_month = rp_month + 1
        for event in events:
            if event.event_type == "postoperative_pathology" and event.month < min_pathology_month:
                event.month = min_pathology_month

        events = sorted(events, key=lambda event: event.month)
        return ProgressionPlan(trajectory=events, final_status=plan.final_status)
