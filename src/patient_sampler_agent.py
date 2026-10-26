from __future__ import annotations

import random

from .base_agent import BaseAgent, compact_json
from .models import PatientProfile


class PatientSamplerAgent(BaseAgent):
    def run(self, archetype: str, knowledge_base, dataset_plan: dict, use_llm: bool = True, rng: random.Random | None = None,
            validation_feedback=None, previous_patient: PatientProfile | None = None) -> PatientProfile:
        rng = rng or random.Random()
        if not use_llm or not self.ollama_client:
            return self._fallback(archetype, rng)
        prompt = self.prompt_template().format(
            archetype=archetype,
            archetype_rules=compact_json(knowledge_base.archetype(archetype)),
            probabilities=compact_json(knowledge_base.get("probabilities_seed")),
            dataset_plan=compact_json(dataset_plan),
        )
        prompt = self.correction_prompt(prompt, validation_feedback, previous_patient)
        raw = self.ollama_client.chat(self.model_name, self.build_messages(prompt), self.temperature, self.num_predict, json_format=True, call_name=self.name)
        return self.parse_json_response(raw, PatientProfile, PatientProfile.model_json_schema())

    def _fallback(self, archetype: str, rng: random.Random) -> PatientProfile:
        palliative = "palliative" in archetype or "mcrpc" in archetype
        watchful = "watchful" in archetype
        age = rng.randint(72, 88) if palliative or watchful else rng.randint(55, 76)
        frailty = "frail" if palliative else ("vulnerable" if age > 75 else "fit")
        return PatientProfile(
            age=age,
            sex="male",
            comorbidity_level="high" if frailty == "frail" else ("moderate" if age > 70 else "low"),
            frailty=frailty,
            life_expectancy_group="<10y" if frailty == "frail" else ("10-15y" if age > 72 else ">15y"),
            family_history_pca=bool(rng.getrandbits(1)),
            brca2_status="positive" if "parp" in archetype or "hrr" in archetype else rng.choice(["unknown", "negative"]),
            presentation_context="Elevacion de PSA en control de salud urologico.",
        )
