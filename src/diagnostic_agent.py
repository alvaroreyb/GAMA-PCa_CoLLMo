from __future__ import annotations

import random

from .base_agent import BaseAgent, compact_json
from .models import Biopsy, Diagnosis, MRI, TNM, PatientProfile


DIAG_BY_RISK = {
    "low_risk": (6.2, "3+3", 1, "cT1c", "cN0", "cM0", "none"),
    "favourable_intermediate": (8.7, "3+4", 2, "cT2a", "cN0", "cM0", "none"),
    "unfavourable_intermediate": (13.4, "4+3", 3, "cT2b", "cN0", "cM0", "none"),
    "high_risk_localised": (24.0, "4+4", 4, "cT2c", "cN0", "cM0", "none"),
    "locally_advanced": (18.0, "4+4", 4, "cT3a", "cN0", "cM0", "none"),
    "cn1_m0": (31.0, "4+5", 5, "cT3b", "cN1", "cM0", "pelvic_nodes"),
    "metastatic_hormone_sensitive_at_diagnosis": (72.0, "4+5", 5, "cT3b", "cN1", "cM1", "bone_and_nodes"),
}


class DiagnosticAgent(BaseAgent):
    def run(self, patient: PatientProfile, archetype: str, knowledge_base, use_llm: bool = True, rng: random.Random | None = None,
            validation_feedback=None, previous_diagnosis: Diagnosis | None = None) -> Diagnosis:
        rng = rng or random.Random()
        if not use_llm or not self.ollama_client:
            return self._fallback(archetype, knowledge_base, rng)
        prompt = self.prompt_template().format(
            patient=compact_json(patient.model_dump()),
            archetype=archetype,
            archetype_rules=compact_json(knowledge_base.archetype(archetype)),
            risk_groups=compact_json(knowledge_base.get("risk_groups")),
            diagnostic_pathways=compact_json(knowledge_base.get("diagnostic_pathways")),
            biopsy_rules=compact_json(knowledge_base.get("biopsy_rules")),
            gleason_isup_mapping=compact_json(knowledge_base.get("gleason_isup_mapping")),
        )
        prompt = self.correction_prompt(prompt, validation_feedback, previous_diagnosis)
        raw = self.ollama_client.chat(self.model_name, self.build_messages(prompt), self.temperature, self.num_predict, json_format=True, call_name=self.name)
        return self.parse_json_response(raw, Diagnosis, Diagnosis.model_json_schema())

    def _fallback(self, archetype: str, kb, rng: random.Random) -> Diagnosis:
        risk = kb.archetype(archetype)["target_risk_groups"][0]
        psa, gleason, isup, ct, cn, cm, sites = DIAG_BY_RISK[risk]
        psa = round(max(0.5, psa * rng.uniform(0.86, 1.18)), 1)
        if "bone_metastases" in archetype:
            sites = "bone"
        if "parp" in archetype:
            sites = "multiple"
        vol = round(rng.uniform(32.0, 68.0), 1)
        lesion_locations = [
            "zona periferica posterolateral derecha",
            "zona periferica izquierda apical",
            "glandula periferica bilateral",
            "zona transicional anterior",
        ]
        dre = "suspicious_cT3" if ct.startswith("cT3") else ("suspicious_cT2" if ct.startswith("cT2") else "normal")
        positive_cores = rng.randint(2, 5) if isup <= 2 else rng.randint(6, 12)
        tumour_extent = "low_volume" if isup <= 1 else ("intermediate_volume" if isup <= 3 else "high_volume")
        extraprostatic_extension = "suspected_on_mri" if ct.startswith(("cT3", "cT4")) else "not_identified"
        return Diagnosis(
            initial_psa=psa,
            repeat_psa=round(psa * rng.uniform(1.03, 1.16), 1),
            dre=dre,
            prostate_volume_cc=vol,
            psa_density=round(psa / vol, 2),
            mri=MRI(performed=True, pirads=5 if isup >= 4 else rng.choice([3, 4]), lesion_location=rng.choice(lesion_locations), local_stage_suspicion=ct),
            biopsy=Biopsy(
                performed=True,
                approach=rng.choice(["transperineal", "transrectal"]),
                strategy="targeted_plus_perilesional",
                histology="adenocarcinoma acinar de prostata",
                gleason=gleason,
                isup=isup,
                positive_cores=positive_cores,
                tumour_extent=tumour_extent,
                cribriform_or_intraductal=isup >= 4,
                lymphovascular_invasion_if_present="not_reported",
                extraprostatic_extension_if_present=extraprostatic_extension,
            ),
            tnm=TNM(cT=ct, cN=cn, cM=cm),
            metastatic_sites=sites,
        )
