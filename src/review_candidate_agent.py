"""One candidate-detection call followed by local verification; no paid repair."""
from pathlib import Path

from .base_agent import compact_json
from .review_candidate_verifier import parse_candidates, verify_candidates


SYSTEM = (Path(__file__).resolve().parent / 'resources/prompts/structured_case_review_prompt_es.txt').read_text(encoding='utf-8').rstrip('\n')


def run_candidate_review(agent, case, context):
    agent.last_candidate_review = None
    agent.last_raw_review = None
    stable = {k: v for k, v in context.items() if k != 'clinical_knowledge/archetypes.yaml'} if agent.cache_friendly_prefix else context
    variable = ('\n\nARQUETIPO AUTORIZADO:\n' + compact_json({'clinical_knowledge/archetypes.yaml': context['clinical_knowledge/archetypes.yaml']})) if agent.cache_friendly_prefix else ''
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': 'CLINICAL_KNOWLEDGE AUTORIZADO:\n' + compact_json(stable)
         + variable + '\n\nCASO JSON:\n' + compact_json(case.model_dump())
         + '\n\nDevuelve solo el JSON FINAL de candidatos.'},
    ]
    raw = agent._logged_response(messages) if agent.reuse_logged_response else None
    if raw is None:
        raw = agent.ollama_client.chat(agent.model_name, messages, agent.temperature,
                                       min(agent.num_predict, agent.concise_num_predict), json_format=True,
                                       call_name=agent.name, reject_truncated=True, max_attempts=1)
    else:
        print('  kimi candidate review: reusing exact logged response; no model call', flush=True)
    agent.last_raw_review = raw
    candidates = parse_candidates(raw)
    review, audit = verify_candidates(candidates, case.model_dump(), context)
    agent.last_candidate_review = audit
    return review
