from types import SimpleNamespace

from src.characterize_batch import write_json
from src.silver.validate import summarize_outputs


def prepare_output(tmp_path, errors=None, bad_offset=False):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (inputs / 'caso_sintetico_2300.txt').write_text('PSA 6 ng/mL', encoding='utf-8')
    model_names = ['bsc', 'medspaner']
    for name in model_names:
        write_json(tmp_path / name / 'patient_0.json', {
            'file_name': 'caso_sintetico_2300.txt', 'models': [name],
            'errors': errors or [], 'entities': [
                {'start': 0, 'end': 99 if bad_offset else 3, 'text': 'PSA', 'label': 'BIOMARKER'}]})
    ann = tmp_path / 'ann'
    ann.mkdir()
    (ann / 'caso_sintetico_2300.ann').write_text('T1\tBIOMARKER 0 3\tPSA\n')
    pipeline = SimpleNamespace(WF1_JSON_DIR=str(tmp_path / 'bsc'), WF2_JSON_DIR=str(tmp_path / 'medspaner'),
                               WF1_MODELS=['bsc'], WF2_MODELS=['medspaner'], ANN_DIR=str(ann),
                               clean_span_text=lambda text: ' '.join(text.split()))
    return pipeline, [{'case_id': 'caso_sintetico_2300'}]

def test_complete_outputs_include_matching_brat_texts_outside_input(tmp_path):
    pipeline, records = prepare_output(tmp_path)
    result = summarize_outputs(tmp_path, pipeline, records)
    assert result['complete']
    assert result['counts']['merged'] == 1
    assert (tmp_path / 'brat/caso_sintetico_2300.txt').read_text() == 'PSA 6 ng/mL'
    assert not (tmp_path / 'inputs/SilverMentions/caso_sintetico_2300.txt').exists()

def test_partial_model_errors_do_not_become_successful_silver(tmp_path):
    pipeline, records = prepare_output(tmp_path, errors=['model: CUDA out of memory'])
    result = summarize_outputs(tmp_path, pipeline, records)
    assert not result['complete']
    assert any('CUDA out of memory' in item for item in result['failures'])

def test_invalid_spans_are_reported(tmp_path):
    pipeline, records = prepare_output(tmp_path, bad_offset=True)
    result = summarize_outputs(tmp_path, pipeline, records)
    assert not result['complete']
    assert any('invalid span' in item for item in result['failures'])
