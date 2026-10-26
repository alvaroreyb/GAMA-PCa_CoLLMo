"""Annotate an isolated snapshot with the existing NER-plus-regex pipeline on CUDA."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import runpy
import shutil

from .characterize_batch import write_json
from .silver.validate import summarize_outputs


PIPELINE = Path(__file__).resolve().parent / 'silver/entidades_all.py'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for NER; CPU fallback is disabled')
    probe = torch.ones((8, 8), device='cuda:0')
    if float((probe @ probe).sum()) != 512:
        raise RuntimeError('CUDA probe failed')
    del probe
    torch.cuda.synchronize()
    return torch


def input_paths(directory):
    directory = Path(directory).resolve(strict=True)
    paths = sorted(directory.glob('*.txt'))
    if not paths:
        raise ValueError('Input directory contains no .txt files')
    for path in paths:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', path.stem):
            raise ValueError('Note filenames must use ASCII letters, digits, underscores or hyphens')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Check input and CUDA without downloading models or writing outputs')
    args = parser.parse_args()
    paths = input_paths(args.input_dir)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    if output == args.input_dir.resolve() or args.input_dir.resolve() in output.parents:
        raise ValueError('NER output must be outside the input directory')
    torch = require_cuda()
    if args.check:
        print(f'CUDA ready: {torch.cuda.get_device_name(0)}; {len(paths)} inputs; no model downloads or writes')
        return
    output.mkdir(parents=True)
    for name in ('inputs', 'original_texts'):
        (output / name).mkdir()
    snapshot = output / 'entidades_all_snapshot.py'
    shutil.copy2(PIPELINE, snapshot)
    status = dict(status='preparing', started_at=datetime.now(timezone.utc).isoformat(),
                  documents=len(paths), device='cuda:0', gpu=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, pipeline_sha256=sha256(snapshot))
    write_json(output / 'run_status.json', status)
    previous_input = os.environ.get('GENEHR_UNTAGGED_DIR')
    os.environ['GENEHR_UNTAGGED_DIR'] = str(output / 'inputs')
    try:
        spec = importlib.util.spec_from_file_location('gama_pca_silver_run', snapshot)
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        records = []
        for patient_id, source in enumerate(paths):
            original = output / 'original_texts' / source.name
            shutil.copy2(source, original)
            raw = original.read_text(encoding='utf-8')
            normalized = pipeline.normalize_input_text(raw)
            paired = output / 'inputs' / source.name
            paired.write_text(normalized, encoding='utf-8')
            records.append(dict(case_id=source.stem, patient_id=patient_id,
                                original_sha256=sha256(original),
                                annotation_text_sha256=sha256(paired),
                                normalization_changed=raw != normalized))
        write_json(output / 'manifest.json', dict(
            documents=len(paths), records=records, pipeline_sha256=sha256(snapshot),
            wf1_models=pipeline.WF1_MODELS, wf2_models=pipeline.WF2_MODELS,
            min_score=pipeline.MIN_SCORE, max_score=pipeline.MAX_SCORE,
            window_stride_tokens=pipeline.WINDOW_STRIDE_TOKENS,
            batch_windows_bsc=pipeline.BATCH_WINDOWS_WF1,
            batch_windows_medspaner=pipeline.BATCH_WINDOWS_WF2))
        status['status'] = 'running'
        write_json(output / 'run_status.json', status)
        runpy.run_path(str(snapshot), run_name='__main__')
        summary = summarize_outputs(output, pipeline, records)
        write_json(output / 'summary.json', summary)
        if not summary['complete']:
            raise RuntimeError('NER output integrity checks failed; see summary.json')
        status.update(status='completed', entities=summary['counts']['merged'],
                      completed_at=datetime.now(timezone.utc).isoformat())
        write_json(output / 'run_status.json', status)
        print(f'Annotated {len(paths)} notes; paired BRAT files: {output / "brat"}')
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'run_status.json', status)
        raise
    finally:
        if previous_input is None:
            os.environ.pop('GENEHR_UNTAGGED_DIR', None)
        else:
            os.environ['GENEHR_UNTAGGED_DIR'] = previous_input


if __name__ == '__main__':
    main()
