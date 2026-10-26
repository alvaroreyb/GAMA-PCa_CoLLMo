"""Operational batch status only; no manuscript analyses or corpus statistics."""
import json
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def characterize(data_dir, start=None, end=None):
    data_dir = Path(data_dir)
    reports = sorted(data_dir.glob('batch_*.json'))
    if start is not None and end is not None:
        reports = [data_dir / f'batch_{start:04d}_{end:04d}.json']
    status = []
    for path in reports:
        if path.is_file():
            report = json.loads(path.read_text())
            status.append({'batch': path.name, **{key: report.get(key, 0) for key in (
                'status', 'planned', 'processed', 'accepted', 'rejected', 'pending_review',
                'awaiting_narrative', 'incomplete_review')}})
    destination = data_dir / 'batch_status.json'
    write_json(destination, {'batches': status})
    return destination
