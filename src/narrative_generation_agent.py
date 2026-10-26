from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from .base_agent import BaseAgent, compact_json
from .models import SyntheticCase


class NarrativeGenerationAgent(BaseAgent):
    def __init__(self, *args, external_prompt_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if external_prompt_path:
            self.prompt_template_path = Path(external_prompt_path)

    def build_prompt(self, case: SyntheticCase) -> str:
        case_json = compact_json(case.model_dump())
        dated_events = self._dated_events(case)
        clinical_flow = "\n".join(
            f"- {event_date.strftime('%d/%m/%Y')}: {event.event_type}; estado={event.state}; "
            f"PSA={event.psa}; tratamiento={event.treatment}; imagen={event.imaging}; notas={event.notes}"
            for event, event_date in dated_events
        )
        prompt = self.prompt_template().format(
            clinical_dossier=self._clinical_dossier(case),
            case_json=case_json,
            scenario=case_json,
            CLINICAL_FLOW=clinical_flow,
            case_index=case.case_id,
        )
        return (
            f"{prompt}\n\n"
            "INSTRUCCIONES ADICIONALES OBLIGATORIAS PARA ESTA EJECUCION:\n"
            "- Redacta como una nota evolutiva hospitalaria real escrita en consulta, no como resumen de dataset.\n"
            "- Usa fechas naturales en formato dd/mm/aaaa, derivadas del flujo clinico. No escribas 'Mes 0', 'Mes 6' ni enumeres eventos como lista cronologica mecanica.\n"
            "- Evita repetir dias y meses exactos; las fechas del dossier ya incluyen variabilidad determinista aproximada de mas/menos 20 dias.\n"
            "- Puede haber estilo humano: abreviaturas clinicas habituales, frases incompletas, alguna omision menor de articulos, puntuacion rapida.\n"
            "- No cometas errores clinicos ni cambies valores numericos.\n"
            "- No uses nombres de claves internas, nombres de arquetipos ni literales en ingles si existe forma clinica en espanol.\n"
            "- Integra los datos en parrafos de evolucion: diagnostico, decision terapeutica, controles, progresion si la hay y situacion actual.\n"
            "- Evita que todas las notas tengan la misma plantilla; varia el orden y la longitud manteniendo cronologia y fidelidad al JSON.\n"
        )

    def run(self, case: SyntheticCase, enabled: bool = True) -> tuple[str, str]:
        prompt = self.build_prompt(case)
        if not enabled:
            return "", prompt
        raw = self.ollama_client.chat(self.model_name, self.build_messages(prompt), self.temperature, self.num_predict, json_format=False, call_name=self.name)
        return raw.strip(), prompt

    def revise(
        self,
        case: SyntheticCase,
        narrative: str,
        reviewer_feedback: str,
    ) -> tuple[str, str]:
        """Revise only the linguistic realisation while preserving the case skeleton."""
        prompt = (
            f"{self.build_prompt(case)}\n\n"
            "REVISION HUMANA OBLIGATORIA:\n"
            "Reescribe la nota incorporando exclusivamente el comentario del revisor. "
            "No modifiques la trayectoria clínica, las fechas, los valores numéricos, "
            "el diagnóstico, el estadio ni los tratamientos del dossier.\n\n"
            f"NOTA ANTERIOR:\n{narrative}\n\n"
            f"COMENTARIO DEL REVISOR:\n{reviewer_feedback}\n"
        )
        raw = self.ollama_client.chat(
            self.model_name,
            self.build_messages(prompt),
            self.temperature,
            self.num_predict,
            json_format=False,
            call_name=f"{self.name}_human_feedback_revision",
        )
        return raw.strip(), prompt

    def _clinical_dossier(self, case: SyntheticCase) -> str:
        events = [
            {
                "fecha": event_date.strftime("%d/%m/%Y"),
                "tipo": event.event_type,
                "estado": event.state,
                "psa": event.psa,
                "testosterona": event.testosterone,
                "imagen": event.imaging,
                "tratamiento": event.treatment,
                "complicacion": event.complication,
                "notas": event.notes,
            }
            for event, event_date in self._dated_events(case)
        ]
        dossier = {
            "paciente": case.patient.model_dump(),
            "diagnostico": case.diagnosis.model_dump(),
            "riesgo": case.risk_classification.model_dump(),
            "tratamiento_inicial": case.treatment_plan.model_dump(),
            "eventos_con_fechas": events,
            "situacion_final": case.progression.final_status.model_dump(),
            "instruccion": "Usar este dossier como fuente cerrada. Redactar nota realista, no plantilla. No usar literales internos ni anglicismos de variables.",
        }
        return compact_json(dossier)

    def _dated_events(self, case: SyntheticCase) -> list[tuple[object, date]]:
        base_date = self._base_date(case.case_id)
        digits = "".join(ch for ch in case.case_id if ch.isdigit())
        case_offset = int(digits or "1")
        dated: list[tuple[object, date]] = []
        previous_date: date | None = None
        for idx, event in enumerate(case.progression.trajectory):
            nominal_date = self._add_months(base_date, event.month)
            day_offset = ((case_offset + idx * 13) % 41) - 20
            event_date = nominal_date + timedelta(days=day_offset)
            if previous_date and event_date <= previous_date:
                event_date = previous_date + timedelta(days=7)
            dated.append((event, event_date))
            previous_date = event_date
        return dated

    @staticmethod
    def _base_date(case_id: str) -> date:
        digits = "".join(ch for ch in case_id if ch.isdigit())
        offset = int(digits or "1")
        year = 2017 + (offset % 5)
        month = 1 + ((offset * 3) % 12)
        day = 3 + ((offset * 7) % 20)
        return date(year, month, day)

    @staticmethod
    def _add_months(value: date, months: int) -> date:
        month_index = value.month - 1 + months
        year = value.year + month_index // 12
        month = month_index % 12 + 1
        day = min(value.day, 28)
        return date(year, month, day)
