from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config_loader import PROJECT_ROOT, output_directory
from .json_utils import parse_json_or_raise, repair_json_with_llm, save_raw_response, validate_pydantic_model


class BaseAgent:
    def __init__(
        self,
        name: str,
        model_name: str | None,
        ollama_client=None,
        prompt_template_path: str | Path | None = None,
        temperature: float = 0.0,
        num_predict: int = 2000,
        raw_output_dir: str | Path | None = None,
    ):
        self.name = name
        self.model_name = model_name
        self.ollama_client = ollama_client
        self.prompt_template_path = Path(prompt_template_path) if prompt_template_path else None
        self.temperature = temperature
        self.num_predict = num_predict
        self.raw_output_dir = Path(raw_output_dir) if raw_output_dir else output_directory() / "raw_llm_outputs"

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def prompt_template(self) -> str:
        if not self.prompt_template_path:
            return ""
        path = self.prompt_template_path
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.read_text(encoding="utf-8")

    def build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "Eres un agente clínico para generar datos sintéticos ficticios. Devuelve JSON estricto si se solicita."},
            {"role": "user", "content": prompt},
        ]

    def parse_json_response(self, raw_text: str, model_cls: type[BaseModel], schema_hint: str) -> BaseModel:
        try:
            return validate_pydantic_model(model_cls, parse_json_or_raise(raw_text))
        except Exception:
            if not self.ollama_client or not self.model_name:
                self.save_raw_output_on_failure(raw_text)
                raise

            repaired = repair_json_with_llm(
                self.ollama_client, self.model_name, raw_text, schema_hint, self.temperature, self.num_predict
            )
            try:
                return validate_pydantic_model(model_cls, parse_json_or_raise(repaired))
            except Exception:
                self.save_raw_output_on_failure(raw_text + "\n\n--- REPAIRED ---\n\n" + repaired)
                raise

    def correction_prompt(self, prompt: str, feedback, previous_block) -> str:
        if not feedback:
            return prompt  # Initial generation remains byte-for-byte unchanged.
        return (
            prompt + "\nCORRECCIÓN LOCALIZADA DEL BLOQUE ANTERIOR:\n"
            "Corrige únicamente los fallos indicados y devuelve el bloque completo en su esquema original. "
            "Conserva los campos válidos; el resto de bloques de entrada son contexto fijo. "
            "No inventes biomarcadores ni cambios basales para justificar un tratamiento.\n"
            "FEEDBACK EXACTO DEL VALIDADOR DETERMINISTA:\n" + "\n".join(feedback)
            + ("\nBLOQUE ANTERIOR:\n" + compact_json(previous_block.model_dump()) if previous_block is not None else "")
        )

    def save_raw_output_on_failure(self, raw_text: str) -> Path:
        return save_raw_response(raw_text, self.raw_output_dir, self.name)


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
