import json
from pathlib import Path

from src.batch_orchestrator_agent import BatchOrchestratorAgent
from src.export_agent import ExportAgent


from src.config_loader import PROJECT_ROOT
ROOT = PROJECT_ROOT


def test_smoke_generate_three_cases_without_narrative(tmp_path):
    orch = BatchOrchestratorAgent(ROOT, use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    orch.dataset_plan["generate_narrative"] = False
    orch.dataset_plan["use_kimi_case_review"] = False
    orch.dataset_plan["use_judge"] = False
    cases = orch.run(total_cases=3)
    assert len(cases) == 3
    assert all(case.case_id.startswith("caso_sintetico_") for case in cases)
    report = json.loads(orch.batch_report_path.read_text())
    assert report["status"] == "completed"
    assert report["processed"] == report["accepted"] == 3
    assert report["rejected"] == 0


def test_narrative_prompt_building(tmp_path):
    orch = BatchOrchestratorAgent(ROOT, use_llm_generation=False, start_index=2300)
    orch.exporter = ExportAgent(tmp_path)
    case = orch._build_case("caso_sintetico_0001", "low_risk_active_surveillance_stable")
    prompt = orch.narrative_agent.build_prompt(case)
    assert "DOSSIER CLINICO FICTICIO" in prompt
    assert "DOSSIER CLINICO:" in prompt
    assert "initial_psa" in prompt
