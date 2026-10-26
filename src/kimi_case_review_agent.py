"""YAML-grounded candidate detection and local verification through Ollama."""
from pathlib import Path

from .base_agent import BaseAgent
from .models import KimiReviewResult, SyntheticCase
from .ollama_client import TruncatedResponseError
import json

REVIEW_KNOWLEDGE_FILES = (
    "archetypes", "biopsy_rules", "diagnostic_pathways", "follow_up_rules",
    "gleason_isup_mapping", "hard_constraints", "progression_rules",
    "risk_groups", "treatment_options",
)


class KimiCaseReviewAgent(BaseAgent):
    concise_review = True
    candidate_review = True
    reuse_logged_response = False
    concise_num_predict = 6000
    cache_friendly_prefix = True

    def run(self, case, knowledge_base, enabled=True):
        if not enabled:
            return KimiReviewResult(passed=True, clinical_plausibility_score=10)
        from .review_candidate_agent import run_candidate_review
        return run_candidate_review(self, case, self._knowledge_context(case, knowledge_base))

    def _logged_response(self, messages):
        """On resume, retry parsing an already paid response to the identical request."""
        trace_dir = getattr(self.ollama_client, "trace_dir", None)
        if not isinstance(trace_dir, (str, Path)):
            return None
        paths = set(Path(trace_dir).glob(f"*_{self.name}.json"))
        for path in sorted(paths, reverse=True):
            try:
                trace = json.loads(path.read_text())
                if (trace.get("model") != self.model_name or trace.get("messages") != messages
                        or trace.get("temperature") != self.temperature
                        or trace.get("num_predict") != min(self.num_predict, self.concise_num_predict)):
                    continue
                response = trace["attempts"][-1]["response"]
                content = response["message"]["content"]
                if response.get("done_reason") == "length":
                    raise TruncatedResponseError(self.model_name, str(content))
                return content if isinstance(content, str) and content.strip() else None
            except (ValueError, KeyError, IndexError, OSError):
                continue
        return None

    def _knowledge_context(self, case: SyntheticCase, knowledge_base) -> dict[str, object]:
        context: dict[str, object] = {}
        for name in REVIEW_KNOWLEDGE_FILES:
            value = knowledge_base.get(name)
            if name == "archetypes":
                value = {"archetypes": {case.archetype: knowledge_base.archetype(case.archetype)}}
            context[f"clinical_knowledge/{name}.yaml"] = value
        return context
