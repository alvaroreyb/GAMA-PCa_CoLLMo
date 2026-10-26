from __future__ import annotations

from .models import Diagnosis, RiskClassification


class RiskClassificationAgent:
    name = "risk_classification"

    def run(self, diagnosis: Diagnosis) -> RiskClassification:
        psa = diagnosis.initial_psa
        isup = diagnosis.biopsy.isup or 0
        ct, cn, cm = diagnosis.tnm.cT, diagnosis.tnm.cN, diagnosis.tnm.cM
        local_ct = ct.startswith("cT1") or ct.startswith("cT2")
        n0 = cn in {"cN0", "cNX"}
        m0 = cm in {"cM0", "cMX"}

        if cm.startswith("cM1"):
            return self._result("metastatic_hormone_sensitive_at_diagnosis", "cM1", "Enfermedad metastásica al diagnóstico.")
        if cn == "cN1" and m0:
            return self._result("cn1_m0", "cN1_cM0", "Afectación ganglionar regional sin metástasis a distancia.")
        if (ct.startswith("cT3") or ct.startswith("cT4")) and m0:
            return self._result("locally_advanced", "cT3_cT4_cM0", "Estadio localmente avanzado sin metástasis.")
        if (isup in {4, 5} or psa > 20) and n0 and m0:
            return self._result("high_risk_localised", "isup_4_5_or_psa_gt_20", "Alto riesgo por ISUP 4/5 o PSA > 20.")
        if isup == 3 and local_ct and n0 and m0:
            return self._result("unfavourable_intermediate", "isup_3_localised", "Riesgo intermedio desfavorable por ISUP 3.")
        if isup == 2 and psa < 10 and local_ct and n0 and m0:
            return self._result("favourable_intermediate", "isup_2_psa_lt_10", "Riesgo intermedio favorable.")
        if isup == 2 and 10 <= psa <= 20 and local_ct and n0 and m0:
            return self._result("unfavourable_intermediate", "isup_2_psa_10_20", "Riesgo intermedio desfavorable por PSA 10-20.")
        if isup == 1 and psa < 10 and local_ct and n0 and m0:
            return self._result("low_risk", "isup_1_psa_lt_10", "Bajo riesgo.")
        if isup == 1 and 10 <= psa <= 20 and local_ct and n0 and m0:
            return self._result("favourable_intermediate", "isup_1_psa_10_20", "Intermedio favorable por PSA 10-20.")
        return self._result("unclassified", "no_rule_matched", "No se encontró una regla determinista aplicable.")

    @staticmethod
    def _result(group: str, rule: str, explanation: str) -> RiskClassification:
        return RiskClassification(risk_group=group, matched_rule=rule, explanation=explanation)
