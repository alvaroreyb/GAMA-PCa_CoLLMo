"""Re-filter saved reviews locally and stage their skeletons for narrative; no API calls."""
import argparse
import fcntl
import json
from pathlib import Path

from .batch_orchestrator_agent import BatchOrchestratorAgent
from .characterize_batch import characterize, write_json
from .models import SyntheticCase
from .review_candidate_verifier import verify_candidates


def recover_reviews(orch, numbers, apply=False):
    data = orch.exporter.output_dir
    with (data / '.generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Existe una generación activa; no se puede recuperar la cola.') from None
        changes, reports = [], {}
        for number in numbers:
            cid = f'caso_sintetico_{number:04d}'
            path = data / 'human_review_queue' / f'{cid}_structured_review.json'
            payload = json.loads(path.read_text())
            if payload.get('status') not in ('pending', 'ready_for_narrative'):
                raise ValueError(f'{cid}: estado no recuperable')
            case = SyntheticCase.model_validate(payload['case'])
            if case.case_id != cid or not orch.validator.run(case).passed:
                raise ValueError(f'{cid}: identidad o validación determinista incorrecta')
            context = orch.kimi_agent._knowledge_context(case, orch.kb)
            candidates = [row['candidate'] for row in payload['verification']['candidates']]
            review, audit = verify_candidates(candidates, case.model_dump(), context)
            if not review.passed:
                raise ValueError(f'{cid}: conserva contradicciones CK confirmadas; no se libera')
            name = payload['batch_report']
            if Path(name).name != name:
                raise ValueError('Ruta de informe inválida')
            report_path = data / name
            report = reports.setdefault(report_path, json.loads(report_path.read_text()))
            outcome = next(row for row in report['outcomes'] if row['case_id'] == cid)
            if outcome['archetype'] != case.archetype or outcome['status'] not in ('pending_review', 'ready_for_narrative'):
                raise ValueError(f'{cid}: resultado del lote no coincide')
            payload.setdefault('original_verification', payload['verification'])
            payload.update(status='ready_for_narrative', verification=audit,
                           automatic_review=review.model_dump(), reviewer_decision=None,
                           resolution='automatic_ck_filter', final_case_exported=False)
            outcome['status'] = 'ready_for_narrative'
            changes.append((path, payload))
        if apply:
            for path, payload in changes:
                write_json(path, payload)
            for path, report in reports.items():
                report['pending_review'] = sum(row['status'] == 'pending_review' for row in report['outcomes'])
                report['awaiting_narrative'] = sum(row['status'] == 'ready_for_narrative' for row in report['outcomes'])
                write_json(path, report)
                start, end = map(int, path.stem.split('_')[1:])
                characterize(data, start, end)
        return [payload['case_id'] for _, payload in changes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-ids', nargs='+', type=int, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    orch = BatchOrchestratorAgent(use_llm_generation=False)
    for cid in recover_reviews(orch, args.case_ids, apply=args.apply):
        print(cid, 'ready_for_narrative' if args.apply else 'would recover; no writes', flush=True)


if __name__ == '__main__':
    main()
