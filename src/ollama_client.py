from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


log = logging.getLogger(__name__)


class TruncatedResponseError(RuntimeError):
    def __init__(self, model: str, content: str):
        super().__init__(f"{model}: response truncated (done_reason=length)")
        self.content = content


@dataclass
class OllamaClient:
    base_url: str
    local_base_url: str | None = None
    cloud_base_url: str | None = None
    timeout_seconds: int = 240
    max_retries: int = 3
    use_json_format: bool = True
    log_llm_calls: bool = False
    log_request_messages: bool = True
    trace_dir: Path | None = None
    sleep_after_call_seconds: float = 15.0
    last_trace: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "OllamaClient":
        ollama = cfg["ollama"]
        return cls(
            base_url=ollama["base_url"].rstrip("/"),
            local_base_url=ollama.get("local_base_url", "").rstrip("/") or None,
            cloud_base_url=ollama.get("cloud_base_url", "").rstrip("/") or None,
            timeout_seconds=int(ollama.get("timeout_seconds", 240)),
            max_retries=int(ollama.get("max_retries", 3)),
            use_json_format=bool(ollama.get("use_json_format", True)),
            sleep_after_call_seconds=float(ollama.get("sleep_after_call_seconds", 15.0)),
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        num_predict: int,
        json_format: bool = False,
        call_name: str = "ollama_chat",
        reject_truncated: bool = False,
        max_attempts: int | None = None,
    ) -> str:
        self.last_trace = None
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if json_format and self.use_json_format:
            payload["format"] = "json"

        last_error: Exception | None = None
        started = time.time()
        request_base_url = self._base_url_for_model(model)
        trace: dict[str, Any] = {
            "call_name": call_name,
            "model": model,
            "base_url": request_base_url,
            "json_format": json_format,
            "temperature": temperature,
            "num_predict": num_predict,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "attempts": [],
        }
        if self.log_request_messages:
            trace["messages"] = messages
        attempt_limit = self.max_retries if max_attempts is None else max_attempts
        if attempt_limit < 1:
            raise ValueError("max_attempts must be positive")
        for attempt in range(1, attempt_limit + 1):
            try:
                attempt_started = time.time()
                resp = requests.post(f"{request_base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                if not str(content).strip():
                    raise RuntimeError(f"Ollama returned empty message content: {json.dumps(data, ensure_ascii=False)[:1200]}")
                trace["attempts"].append(
                    {
                        "attempt": attempt,
                        "status": "ok",
                        "duration_seconds": round(time.time() - attempt_started, 3),
                        "response": data,
                    }
                )
                trace["duration_seconds"] = round(time.time() - started, 3)
                trace["content_preview"] = str(content)[:1000]
                trace["usage"] = {
                    "input_tokens": data.get("prompt_eval_count"),
                    "cached_input_tokens": data.get("prompt_eval_cached_count"),
                    "output_tokens": data.get("eval_count"),
                }
                self._write_trace(trace)
                if reject_truncated and data.get("done_reason") == "length":
                    raise TruncatedResponseError(model, str(content))
                if self.sleep_after_call_seconds > 0 and not self._is_cloud_model(model):
                    time.sleep(self.sleep_after_call_seconds)
                return content
            except TruncatedResponseError:
                # Already logged with usage. Do not replay the same request or
                # attempt JSON repair on a response known to be incomplete.
                raise
            except Exception as exc:
                last_error = exc
                trace["attempts"].append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "duration_seconds": round(time.time() - started, 3),
                        "error": str(exc),
                    }
                )
                log.warning("Ollama chat failed on attempt %s/%s: %s", attempt, attempt_limit, exc)
                if attempt < attempt_limit:
                    time.sleep(min(2 ** attempt, 8))
        trace["duration_seconds"] = round(time.time() - started, 3)
        trace["final_error"] = str(last_error)
        self._write_trace(trace)
        raise RuntimeError(f"Ollama chat failed after {attempt_limit} attempts: {last_error}")

    def _base_url_for_model(self, model: str) -> str:
        if self._is_cloud_model(model):
            return (self.cloud_base_url or self.base_url).rstrip("/")
        return (self.local_base_url or self.base_url).rstrip("/")

    @staticmethod
    def _is_cloud_model(model: str) -> bool:
        name = model.strip().lower()
        return name.endswith(":cloud") or name.endswith("-cloud")


    def _write_trace(self, trace: dict[str, Any]) -> None:
        self.last_trace = trace
        if not self.log_llm_calls or not self.trace_dir:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in trace["call_name"])
        path = self.trace_dir / f"{stamp}_{safe_name}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
