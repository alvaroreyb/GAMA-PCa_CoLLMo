"""Integrity checks for the existing NER-plus-regex exports."""
from collections import Counter
import json
import shutil
from pathlib import Path


def summarize_outputs(output, pipeline, records):
    counts, failures, labels = Counter(), [], Counter()
    for workflow, json_dir, models in [
        ('BSC', Path(pipeline.WF1_JSON_DIR), pipeline.WF1_MODELS),
        ('MedSpaner', Path(pipeline.WF2_JSON_DIR), pipeline.WF2_MODELS),
    ]:
        files = sorted(json_dir.glob('patient_*.json'))
        if len(files) != len(records):
            failures.append(f'{workflow}: {len(files)} JSON files for {len(records)} cases')
        for index, record in enumerate(records):
            path = json_dir / f'patient_{index}.json'
            if not path.exists():
                failures.append(f'{workflow}: missing {record["case_id"]}')
                continue
            result = json.loads(path.read_text())
            if result['file_name'] != record['case_id'] + '.txt' or result['models'] != models:
                failures.append(f'{workflow}: incorrect identity/model list at patient_{index}')
            failures.extend(f'{workflow}/{record["case_id"]}: {error}' for error in result.get('errors', []))
            text = (output / 'inputs' / result['file_name']).read_text()
            for entity in result.get('entities', []):
                start, end = entity['start'], entity['end']
                if not (0 <= start < end <= len(text)) or pipeline.clean_span_text(text[start:end]) != entity['text']:
                    failures.append(f'{workflow}/{record["case_id"]}: invalid span {start}:{end}')
                counts[workflow] += 1
                labels[entity['label']] += 1
    brat = output / 'brat'
    brat.mkdir()
    for record in records:
        cid = record['case_id']
        ann = Path(pipeline.ANN_DIR) / f'{cid}.ann'
        text_path = output / 'inputs' / f'{cid}.txt'
        if not ann.exists():
            failures.append(f'Missing merged Brat annotation: {cid}')
            continue
        text = text_path.read_text()
        for line in ann.read_text().splitlines():
            if not line:
                continue
            _, span, mention = line.split('\t', 2)
            label, start, end = span.split()
            start, end = int(start), int(end)
            if not (0 <= start < end <= len(text)) or pipeline.clean_span_text(text[start:end]) != pipeline.clean_span_text(mention):
                failures.append(f'{cid}: invalid merged span {start}:{end}')
            counts['merged'] += 1
            counts[f'merged:{label}'] += 1
        shutil.copy2(ann, brat / ann.name)
        shutil.copy2(text_path, brat / text_path.name)
    return dict(counts=counts, ner_labels=labels, failures=failures,
                documents=len(records), complete=not failures)
