from __future__ import annotations

from pathlib import Path

from .base_agent import BaseAgent, compact_json
from .models import JudgeResult, SyntheticCase


class JudgeAgent(BaseAgent):
    def __init__(self, *args, external_prompt_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if external_prompt_path:
            self.prompt_template_path = Path(external_prompt_path)

    def run(self, case: SyntheticCase, narrative: str, enabled: bool = True) -> JudgeResult | None:
        if not enabled:
            return None
        messages = self.build_prompt(case, narrative)
        raw = self.ollama_client.chat(
            self.model_name,
            messages,
            self.temperature,
            self.num_predict,
            json_format=True,
            call_name=self.name,
        )
        return self.parse_json_response(
            raw,
            JudgeResult,
            compact_json(JudgeResult.model_json_schema()),
        )

    def build_prompt(self, case: SyntheticCase, narrative: str) -> list[dict[str, str]]:
        # The published method is narrative-only: the structured skeleton is
        # deliberately not exposed to the judge.
        del case
        prompt = self.prompt_template().replace("{narrative}", narrative)
        return self.build_messages(prompt)
