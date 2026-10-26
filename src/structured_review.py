"""Explicit human clearance of a queued skeleton, followed by normal narrative checks."""
import fcntl
import json
from datetime import datetime, timezone

from .batch_orchestrator_agent import BatchOrchestratorAgent
from .characterize_batch import characterize, write_json
from .config_loader import PROJECT_ROOT, output_directory
from .export_agent import build_metadata
from .models import SyntheticCase, KimiReviewResult


def resolve_structured_review(case_id, feedback='', root=PROJECT_ROOT):
    data_dir = output_directory(root)
    path = data_dir / 'human_review_queue' / f'{case_id}_structured_review.json'
    with (data_dir / '.generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Hay una generación activa; revisa el esqueleto cuando termine.') from None
        payload = json.loads(path.read_text())
        if payload.get('final_case_exported'):
            return path
        if payload['verification']['confirmed']:
            raise ValueError('Hay contradicciones confirmadas: deben corregirse antes de aceptar el esqueleto.')
        case = SyntheticCase.model_validate(payload['case'])
        report_path = data_dir / payload['batch_report']
        if report_path.parent != data_dir or not report_path.name.startswith('batch_'):
            raise ValueError('Ruta de lote inválida')
        report = json.loads(report_path.read_text())
        outcome = next(row for row in report['outcomes'] if row['case_id'] == case_id)
        if outcome['status'] != 'pending_review':
            raise ValueError('El caso no está pendiente en el informe del lote')
        orch = BatchOrchestratorAgent(root=root, use_llm_generation=True)
        deterministic = orch.validator.run(case)
        if not deterministic.passed:
            raise ValueError(f'El esqueleto ya no pasa el determinista: {deterministic.errors}')
        # Human clearance is explicit, not a fabricated Kimi approval.
        payload.update(reviewer_decision='accept', reviewer_feedback=feedback,
                       reviewed_at=datetime.now(timezone.utc).isoformat(), status='human_cleared')
        write_json(path, payload)
        case.metadata['structured_review_human_clearance'] = {
            'queue': path.name, 'reviewed_at': payload['reviewed_at'], 'feedback': feedback}
        orch._artifact_prefix = 'human_structured_clearance_'
        limit = int(orch.dataset_plan.get('max_narrative_attempts_per_case', 3))
        result = orch._run_narrative_attempts(case, case_id, limit)
        if result is None:
            payload['status'] = 'narrative_failed_after_clearance'
            write_json(path, payload)
            raise ValueError('Narrativa no aceptada; esqueleto conservado. No se ha aprobado el caso.')
        narrative, prompt, judge = result
        kimi = KimiReviewResult(passed=False, warnings=['Revisión de Kimi pendiente resuelta explícitamente por humano; ver cola.'])
        metadata = build_metadata(case, deterministic, kimi, judge)
        orch.exporter.export_case(case, narrative, prompt, metadata, kimi, judge)
        outcome['status'] = 'accepted'
        outcome['review_resolution'] = 'human_clearance'
        report['pending_review'] = sum(r['status'] == 'pending_review' for r in report['outcomes'])
        report['accepted'] = sum(r['status'] == 'accepted' for r in report['outcomes'])
        report['accepted_by_archetype'][case.archetype] += 1
        write_json(report_path, report)
        payload.update(status='accepted_by_human', final_case_exported=True)
        write_json(path, payload)
        start, end = (int(v) for v in report_path.stem.split('_')[1:])
        characterize(data_dir, start, end)
        return path
