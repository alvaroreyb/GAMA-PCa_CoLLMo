"""Closed-world CK filter: only proven contradictions block downstream generation."""
from __future__ import annotations

import json
import re

from .models import KimiReviewResult


class PendingReviewCandidatesError(ValueError):
    pass


def resolve(node, path):
    if not path:
        raise ValueError('Empty evidence path')
    for part in path.split('.'):
        if isinstance(node, list) and part.isdigit():
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node[part]
        else:
            raise ValueError('Unresolvable evidence path')
    return node


def same_literal(left, right):
    """JSON equality without Python's True == 1 coercion."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_literal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_literal(a, b) for a, b in zip(left, right))
    return left == right


def documented_grade(value):
    """Accept only an entire, unambiguous scalar/grade label, not prose."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1 <= value <= 5 else None
    if isinstance(value, str):
        match = re.fullmatch(r'\s*(?:(?:ISUP|Grupo\s+de\s+grado|Grade\s+Group)\s*[:=]?\s*)?([1-5])\s*', value, re.I)
        if not match:
            match = re.fullmatch(r'\s*Grado\s+([1-5])\s+de\s+ISUP\s*', value, re.I)
        if match:
            return int(match.group(1))
    return None


def parse_candidates(raw):
    text = raw.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {'candidates'} or not isinstance(data['candidates'], list):
        raise ValueError('Expected only candidates array, not a verdict/checklist')
    fields = {'case_path', 'rule_path', 'condition', 'yaml_value', 'case_value'}
    for item in data['candidates']:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError('Incomplete candidate evidence contract')
        if any(not isinstance(item[k], str) or not item[k].strip() for k in ('case_path', 'rule_path', 'condition')):
            raise ValueError('Invalid candidate evidence paths/condition')
    return data['candidates']


def applicability(rule, case):
    diagnosis = case['diagnosis']
    values = {'risk_group': case['risk_classification']['risk_group'],
              'psa': diagnosis['initial_psa'], 'isup': diagnosis['biopsy']['isup'],
              'metastatic_sites': diagnosis['metastatic_sites'],
              'treatment': case['treatment_plan']['initial_treatment'], **diagnosis['tnm']}
    results = []
    for key, expected in rule.get('if', {}).items():
        if key.endswith('_in'):
            actual = values.get(key[:-3])
            results.append(None if actual is None else actual in expected)
        elif key.endswith('_not'):
            actual = values.get(key[:-4])
            results.append(None if actual is None else actual != expected)
        elif key.endswith('_gt'):
            actual = values.get(key[:-3])
            results.append(None if actual is None else actual > expected)
        elif key in values:
            results.append(values[key] == expected)
        else:
            results.append(None)
    return False if False in results else None if None in results else True


def _check(item, case, context):
    citation = item['rule_path'].strip('[]')
    file, path = citation.split(':', 1)
    yaml_value = resolve(context[file], path)
    # Exact literal snapshots: a claimed numeric grade in text is handled below.
    if not same_literal(item['yaml_value'], yaml_value):
        return 'pending', 'YAML_VALUE no coincide con el valor de la ruta citada'
    actual = resolve(case, item['case_path'])
    if file == 'clinical_knowledge/gleason_isup_mapping.yaml' and path.startswith('gleason_to_isup.'):
        gleason = path.split('.', 1)[1]
        if item['case_path'] == 'diagnosis.biopsy.gleason':
            if not same_literal(actual, item['case_value']) or actual != gleason:
                return 'pending', 'Gleason citado no coincide con la biopsia/ruta YAML'
            grade = case['diagnosis']['biopsy']['isup']
            return ('confirmed' if grade != yaml_value else 'not_violated'), f'Biopsia Gleason {gleason}: ISUP={grade}, YAML={yaml_value}'
        if isinstance(actual, str):
            gleasons = {a + '+' + b for a, b in re.findall(r'\bGleason\s*[:=]?\s*([3-5])\s*\+\s*([3-5])', actual, re.I)}
            grades = {int(v) for v in re.findall(r'\bISUP\s*(?:grupo\s*)?[:=]?\s*([1-5])\b', actual, re.I)}
            grades.update(int(v) for v in re.findall(r'\bgrado\s*([1-5])\s*de\s*ISUP\b', actual, re.I))
            grades.update(int(v) for v in re.findall(r'\b(?:Grade\s+Group|Grupo\s+de\s+grado)\s*[:=]?\s*([1-5])\b', actual, re.I))
            if gleasons != {gleason} or len(grades) != 1:
                return 'pending', 'No se puede vincular inequívocamente Gleason e ISUP en el texto citado'
            actual = next(iter(grades))
        elif item['case_path'] == 'diagnosis.biopsy.isup':
            if case['diagnosis']['biopsy']['gleason'] != gleason:
                return 'pending', 'El Gleason de la regla no corresponde a la biopsia citada'
        else:
            return 'pending', 'Campo no vinculado al mapeo Gleason–ISUP'
        if not same_literal(actual, documented_grade(item['case_value'])):
            return 'pending', 'CASE_VALUE no coincide con el grado documentado'
        return ('confirmed' if actual != yaml_value else 'not_violated'), f'Gleason {gleason}: ISUP documentado={actual}, YAML={yaml_value}'
    if not same_literal(actual, item['case_value']):
        return 'pending', 'CASE_VALUE no coincide con el campo citado'

    if file == 'clinical_knowledge/biopsy_rules.yaml' and path.startswith('pathology_required_fields'):
        if actual is None:
            return 'not_violated', 'La lista exige campos, pero no contiene una prohibición literal de null'
        return 'pending', 'Requisito patológico no verificable con este candidato'

    if file == 'clinical_knowledge/hard_constraints.yaml':
        parts = path.split('.')
        if len(parts) < 2 or parts[0] != 'hard_constraints' or not parts[1].isdigit():
            return 'pending', 'Debe citarse la ruta original con índice de hard_constraints'
        rule = resolve(context[file], '.'.join(parts[:2]))
        applies = applicability(rule, case)
        if applies is False:
            return 'not_violated', 'La condición de aplicación YAML es falsa para el caso'
        if applies is None:
            return 'pending', 'Condición de aplicación no ejecutable'
        if rule['id'] == 'psa_density_consistency' and item['case_path'] == 'diagnosis.psa_density':
            d = case['diagnosis']
            expected = d['initial_psa'] / d['prostate_volume_cc']
            tolerance = max(rule['tolerance_abs'], abs(expected) * rule['tolerance_rel'])
            violated = abs(actual - expected) > tolerance
            return ('confirmed' if violated else 'not_violated'), f'densidad={actual}; calculada={expected}; tolerancia={tolerance}'
        if rule['id'] == 'events_sorted_by_month':
            months = [event['month'] for event in case['progression']['trajectory']]
            return ('confirmed' if months != sorted(months) else 'not_violated'), f'meses={months}'
        return 'pending', 'Regla dura sin verificador local específico'

    if file == 'clinical_knowledge/progression_rules.yaml':
        parts = path.split('.')
        match = re.match(r'^progression\.trajectory\.(\d+)(?:\.|$)', item['case_path'])
        if len(parts) < 3 or parts[0] != 'event_definitions' or not match:
            return 'pending', 'No se ha vinculado regla de evento con un evento concreto'
        index = int(match.group(1))
        events = case['progression']['trajectory']
        event = events[index]
        if event['event_type'] != parts[1]:
            return 'pending', 'La regla no corresponde al tipo del evento citado'
        rule = resolve(context[file], '.'.join(parts[:2]))
        key = parts[2]
        previous = {v for ev in events[:index] for v in (ev['event_type'], ev.get('treatment')) if v}
        if key in ('required_previous_events', 'required_previous_events_any'):
            required = set(rule[key])
            violated = (bool(required) and not required.intersection(previous)) if key.endswith('_any') else not required.issubset(previous)
            return ('confirmed' if violated else 'not_violated'), f'requeridos={sorted(required)}; previos={sorted(previous)}'
        if key in ('allowed_risk_groups', 'possible_treatments'):
            value = case['risk_classification']['risk_group'] if key == 'allowed_risk_groups' else event.get('treatment')
            return ('confirmed' if value not in rule[key] else 'not_violated'), f'valor={value}; permitidos={rule[key]}'
        if key in ('psa_min', 'psa_max', 'psa_range') and event.get('psa') is not None:
            psa = event['psa']
            violated = psa < rule[key] if key == 'psa_min' else psa > rule[key] if key == 'psa_max' else not rule[key][0] <= psa <= rule[key][1]
            return ('confirmed' if violated else 'not_violated'), f'PSA={psa}; regla={rule[key]}'
        return 'pending', 'Condición de progresión no ejecutable por este verificador'
    if file == 'clinical_knowledge/treatment_options.yaml' and path.startswith('allowed_treatments_by_risk_group.'):
        parts = path.split('.')
        if parts[1] != case['risk_classification']['risk_group'] or item['case_path'] != 'treatment_plan.initial_treatment':
            return 'pending', 'Lista de tratamientos no vinculada al riesgo/tratamiento inicial'
        group = context[file]['allowed_treatments_by_risk_group'][parts[1]]
        allowed = {v for key in ('preferred', 'alternatives', 'selected_cases') for v in group.get(key, [])}
        return ('confirmed' if actual not in allowed else 'not_violated'), f'tratamiento={actual}; permitidos completos={sorted(allowed)}'
    return 'pending', 'Hallazgo fuera de la cobertura local; requiere revisión, no aprobación automática'


def verify_candidates(candidates, case, context):
    rows, errors, warnings = [], [], []
    for item in candidates:
        try:
            status, explanation = _check(item, case, context)
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
            status, explanation = 'pending', f'Evidencia/ruta no verificable: {exc}'
        if status == 'pending':
            status = 'unsupported'
        rows.append({'candidate': item, 'status': status, 'local_evidence': explanation})
        if status == 'confirmed':
            errors.append(f"[{item['rule_path'].strip('[]')}] YAML_VALUE={json.dumps(item['yaml_value'], ensure_ascii=False)} "
                          f"CASE_VALUE={json.dumps(item['case_value'], ensure_ascii=False)} {explanation}")
        elif status == 'unsupported':
            warnings.append(explanation)
    audit = {'version': 2, 'policy': 'only_confirmed_ck_contradictions_block',
             'candidates': rows, 'pending': 0, 'unsupported': len(warnings),
             'confirmed': len(errors), 'not_violated': sum(row['status'] == 'not_violated' for row in rows)}
    # Compatibility score is not an LLM judgment; it is not used as the judge rubric.
    review = KimiReviewResult(passed=not errors, errors=errors, warnings=warnings)
    return review, audit
