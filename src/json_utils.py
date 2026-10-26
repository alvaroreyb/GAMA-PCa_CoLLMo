from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> str:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
            return text[idx : idx + end]
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object or array found in response")


def parse_json_or_raise(text: str) -> Any:
    return json.loads(extract_json_from_text(text))


def validate_pydantic_model(model_cls: type[T], data: Any) -> T:
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def save_raw_response(text: str, output_dir: str | Path, prefix: str = "raw") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    path = out / f"{prefix}_{stamp}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def repair_json_with_llm(ollama_client, model: str, raw_text: str, schema_hint: str, temperature: float, num_predict: int) -> str:
    messages = [
        {"role": "system", "content": "Repara respuestas para que sean JSON estricto. Devuelve solo JSON válido."},
        {
            "role": "user",
            "content": (
                "Convierte esta respuesta en un objeto JSON válido que respete el esquema indicado. "
                "No añadas claves ni explicaciones.\n\n"
                f"ESQUEMA:\n{schema_hint}\n\nRESPUESTA:\n{raw_text}"
            ),
        },
    ]
    return ollama_client.chat(model, messages, temperature, num_predict, json_format=True, call_name="json_repair")
