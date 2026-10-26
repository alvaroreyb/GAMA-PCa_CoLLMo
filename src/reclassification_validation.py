"""Validate an explicitly declared AS reclassification against the original YAML.

Free-text extraction is deliberately conservative: ambiguous or unsupported
evidence yields a warning, never an invented stage or a clinical rejection.
"""
from __future__ import annotations

import re
import unicodedata


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')


def match_rule(rule, state):
    """Three-valued evaluation of the CK's all/any predicates."""
    values = []
    for key, value in rule.items():
        if key in ('all', 'any'):
            children = value if isinstance(value, list) else [value]
            results = [match_rule(child, state) for child in children]
            if key == 'all':
                result = False if False in results else None if None in results else True
            else:
                result = True if True in results else None if None in results else False
        else:
            field, _, op = key.rpartition('_')
            actual = state.get(field)
            if actual is None:
                result = None
            elif op == 'in':
                result = actual in value
            elif op in ('lt', 'lte', 'gt', 'gte') and isinstance(actual, (int, float)):
                result = {'lt': actual < value, 'lte': actual <= value,
                          'gt': actual > value, 'gte': actual >= value}[op]
            else:
                result = None
        values.append(result)
    return False if False in values else None if None in values else True


def classify(state, knowledge):
    groups = knowledge['risk_groups']
    order = knowledge.get('classification_order') or sorted(groups, key=lambda g: groups[g]['priority'])
    for group in order:
        rule = {k: v for k, v in groups[group].items() if k in ('all', 'any')}
        if not rule:
            return None
        matches = match_rule(rule, state)
        if matches is None:
            return None  # An unresolved higher-priority rule can change the result.
        if matches:
            return group
    return None


def declared_groups(text, knowledge):
    text = normalized(text)
    found = set()
    for group, rule in knowledge['risk_groups'].items():
        aliases = [group, rule.get('label_es', '')]
        if group in ('favourable_intermediate', 'unfavourable_intermediate'):
            aliases.append('intermedio ' + ('desfavorable' if group.startswith('un') else 'favorable'))
        for alias in aliases:
            if alias and re.search(r'(?<!\w)' + re.escape(normalized(alias)) + r'(?!\w)', text):
                found.add(group)
    return found


def validate_reclassifications(case, knowledge_base):
    knowledge = knowledge_base.get('risk_groups')
    mapping = knowledge_base.gleason_to_isup()
    diagnosis = case.diagnosis
    state = {'isup': diagnosis.biopsy.isup, 'psa': diagnosis.initial_psa,
             **diagnosis.tnm.model_dump()}
    sources = {key: 'diagnosis' for key in state}
    errors, warnings = [], []
    # Input ordering is independently validated. Never use future evidence.
    events = sorted(enumerate(case.progression.trajectory), key=lambda pair: (pair[1].month, pair[0]))
    for index, event in events:
        notes = event.notes or ''
        source = f'trajectory[{index}] mes={event.month}'
        # Clinical TNM in prior/current notes can update the known stage. Do not
        # infer cT from MRI descriptions or use postoperative pT/pN as clinical TNM.
        stage_text = notes + ' ' + (event.systemic_staging_result or '')
        stage_text = re.sub(r'(c?T[1-4][abc]?)(?=c?N[01X])', r'\1 ', stage_text, flags=re.IGNORECASE)
        stage_text = re.sub(r'(c?N[01X])(?=c?M[01X])', r'\1 ', stage_text, flags=re.IGNORECASE)
        for field, pattern in (
            ('cT', r'(?<![a-zA-Z])c?T([1-4](?:[abc])?|X)(?:(?![a-zA-Z0-9])|(?=c?N[01X]))'),
            ('cN', r'(?<![a-zA-Z])c?N([01X])(?:(?![a-zA-Z0-9])|(?=c?M[01X]))'),
            ('cM', r'(?<![a-zA-Z])c?M([01](?:[abc])?|X)(?![a-zA-Z0-9])'),
        ):
            values = {field + (v.upper() if v.lower() == 'x' else v.lower())
                      for v in re.findall(pattern, stage_text, re.IGNORECASE)}
            if values:
                state[field] = next(iter(values)) if len(values) == 1 else None
                sources[field] = source
        grades = {int(v) for v in re.findall(r'\bISUP\s*(?:grupo\s*)?[:=]?\s*([1-5])\b', notes, re.IGNORECASE)}
        grades.update(int(v) for v in re.findall(r'\bgrado\s*([1-5])\s*de\s*ISUP\b', notes, re.IGNORECASE))
        grades.update(int(v) for v in re.findall(r'\b(?:Grade\s+Group|Grupo\s+de\s+grado)\s*[:=]?\s*([1-5])\b', notes, re.IGNORECASE))
        gleasons = re.findall(r'\bGleason\s*[:=]?\s*([3-5])\s*\+\s*([3-5])', notes, re.IGNORECASE)
        for a, b in gleasons:
            grades.add(mapping.get(a + '+' + b))
        if grades:
            state['isup'] = next(iter(grades)) if len(grades) == 1 else None
            sources['isup'] = source
        if event.psa is not None:
            state['psa'] = event.psa
            sources['psa'] = source
        else:
            psa_values = {float(value.replace(',', '.')) for value in re.findall(
                r'\bPSA\s*(?:(?:de|a)\s+|[:=]\s*)?(\d+(?:[.,]\d+)?)', notes, re.IGNORECASE)}
            if psa_values:
                state['psa'] = next(iter(psa_values)) if len(psa_values) == 1 else None
                sources['psa'] = source + ' notes'
        if event.event_type != 'active_surveillance_reclassification':
            continue
        declared = declared_groups(notes + ' ' + event.state, knowledge)
        if not declared:
            continue  # Do not introduce a mandatory risk-label field.
        expected = classify(state, knowledge)
        if len(declared) != 1 or expected is None:
            warnings.append(f'reclassification_risk_unverifiable: {source}; datos o etiqueta ambiguos; '
                            'no se exige repetir TNM previamente documentado')
            continue
        actual = next(iter(declared))
        if actual != expected:
            evidence = ', '.join(f'{key}={state[key]} ({sources[key]})' for key in state)
            errors.append(
                f'reclassification_risk_consistency: {source}; declarado={actual}; esperado={expected}; '
                f'{evidence}; reglas=clinical_knowledge/risk_groups.yaml:risk_groups.{expected} '
                f'y risk_groups.{actual}. Corrige solo la etiqueta/notas inconsistentes de progresión '
                'conservando los datos documentados; no cambies PSA, ISUP o TNM para justificar la etiqueta. '
                'No regeneres paciente, diagnóstico, riesgo inicial ni tratamiento inicial.'
            )
    return errors, warnings
