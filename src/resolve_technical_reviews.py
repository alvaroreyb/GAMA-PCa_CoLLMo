"""Apply explicit, case-bound assisted CK decisions; never calls any model."""
import argparse
import fcntl
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .batch_orchestrator_agent import BatchOrchestratorAgent
from .characterize_batch import characterize, write_json
from .models import SyntheticCase, KimiReviewResult
from .review_candidate_verifier import verify_candidates


def case_digest(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def complete_candidate_prefix(raw):
    """Extract only finished objects for audit; never complete or repair a truncation."""
    match = re.search(r'"candidates"\s*:\s*\[', raw)
    if not match:
        return []
    decoder, rows, pos = json.JSONDecoder(), [], match.end()
    while pos < len(raw):
        pos += len(raw[pos:]) - len(raw[pos:].lstrip(' \r\n\t,'))
        try:
            row, end = decoder.raw_decode(raw[pos:])
        except ValueError:
            break
        if not isinstance(row, dict):
            break
        rows.append(row)
        pos += end
    return rows


def apply_decisions(orch, decisions, apply=False):
    data = orch.exporter.output_dir
    changes, reports = [], {}
    with (data / '.generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Generación activa; no se modifica la cola.') from None
        for entry in decisions:
            cid = entry['case_id']
            if not re.fullmatch(r'caso_sintetico_\d+', cid):
                raise ValueError('Identificador inválido')
            path = data / 'technical_review_queue' / f'{cid}_incomplete_review.json'
            payload = json.loads(path.read_text())
            if payload.get('status') not in ('incomplete_review', 'awaiting_user', 'ready_for_narrative'):
                raise ValueError(f'{cid}: estado no resoluble')
            if case_digest(payload['case']) != entry['case_sha256']:
                raise ValueError(f'{cid}: el esqueleto cambió tras la auditoría')
            case = SyntheticCase.model_validate(payload['case'])
            if case.case_id != cid:
                raise ValueError('Identidad del caso no coincide')
            if not entry.get('reason') or entry['decision'] not in ('approve', 'hold'):
                raise ValueError('Falta decisión/justificación explícita')
            name = payload['batch_report']
            if Path(name).name != name:
                raise ValueError('Ruta de lote inválida')
            report_path = data / name
            report = reports.setdefault(report_path, json.loads(report_path.read_text()))
            outcome = next(r for r in report['outcomes'] if r['case_id'] == cid)
            if outcome['status'] not in ('incomplete_review', 'ready_for_narrative') or outcome['archetype'] != case.archetype:
                raise ValueError(f'{cid}: posición de lote incoherente')
            deterministic = orch.validator.run(case)
            context = json.loads(json.dumps(orch.kimi_agent._knowledge_context(case, orch.kb)))
            _, audit = verify_candidates(complete_candidate_prefix(payload.get('raw_response', '')), case.model_dump(), context)
            if entry['decision'] == 'approve' and (not deterministic.passed or audit['confirmed']):
                raise ValueError(f'{cid}: contradicción ejecutable; no liberar')
            payload['assisted_review'] = dict(entry, reviewer=entry.get('reviewer', 'human'), model_calls=0,
                                             reviewed_at=datetime.now(timezone.utc).isoformat(),
                                             llm_review_complete=False, prefix_audit=audit)
            if entry['decision'] == 'approve':
                payload.update(status='ready_for_narrative', final_case_exported=False,
                               automatic_review=KimiReviewResult(passed=True, warnings=[
                                   'Liberado mediante auditoría independiente CK; Kimi no completó la revisión.']).model_dump())
                outcome.update(status='ready_for_narrative', review_resolution='assisted_ck_audit',
                               recovery_queue=str(path.relative_to(data)))
            else:
                payload['status'] = 'awaiting_user'
            changes.append((path, payload))
        if apply:
            for path, payload in changes:
                write_json(path, payload)
            for path, report in reports.items():
                report['incomplete_review'] = sum(r['status'] == 'incomplete_review' for r in report['outcomes'])
                report['awaiting_narrative'] = sum(r['status'] == 'ready_for_narrative' for r in report['outcomes'])
                write_json(path, report)
                start, end = map(int, path.stem.split('_')[1:])
                characterize(data, start, end)
    return [(p['case_id'], p['status']) for _, p in changes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    orch = BatchOrchestratorAgent(use_llm_generation=False)
    for cid, status in apply_decisions(orch, json.loads(args.manifest.read_text())['decisions'], args.apply):
        print(cid, status, 'saved' if args.apply else 'dry run')


if __name__ == '__main__':
    main()
