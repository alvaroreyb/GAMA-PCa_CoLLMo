from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config_loader import output_directory
from .characterize_batch import write_json
from .models import CaseMetadata, JudgeResult, KimiReviewResult, SyntheticCase, ValidationResult


class ExportAgent:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else output_directory()
        for sub in (
            "generated_json",
            "generated_txt",
            "generated_prompts",
            "generated_metadata",
            "generated_artifacts",
            "judge_reports",
            "human_review_queue",
            "technical_review_queue",
            "rejected_cases",
        ):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "dataset_index.csv"

    def export_case(
        self,
        case: SyntheticCase,
        narrative: str,
        narrative_prompt: str,
        metadata: CaseMetadata,
        kimi: KimiReviewResult | None,
        judge: JudgeResult | None,
    ) -> None:
        cid = case.case_id
        self._write_json(self.output_dir / "generated_json" / f"{cid}.json", case.model_dump())
        (self.output_dir / "generated_txt" / f"{cid}.txt").write_text(narrative, encoding="utf-8")
        (self.output_dir / "generated_prompts" / f"{cid}_prompt.txt").write_text(narrative_prompt, encoding="utf-8")
        self._write_json(self.output_dir / "generated_metadata" / f"{cid}.meta.json", metadata.model_dump())
        if kimi:
            self._write_json(self.output_dir / "generated_metadata" / f"{cid}_kimi_review.json", kimi.model_dump())
        if judge:
            self._write_json(self.output_dir / "judge_reports" / f"{cid}_judge.json", judge.model_dump())
            if judge.requires_revision:
                self._write_json(
                    self.output_dir / "human_review_queue" / f"{cid}_review.json",
                    {
                        "case_id": cid,
                        "status": "pending",
                        "narrative": narrative,
                        "judge": judge.model_dump(),
                        "reviewer_decision": None,
                        "reviewer_feedback": None,
                    },
                )
        self.append_index(metadata)

    def export_rejected(self, case_id: str, payload: dict[str, Any]) -> None:
        path = self.output_dir / "rejected_cases" / f"caso_rechazado_{case_id}.json"
        if path.exists():
            idx = 2
            while True:
                candidate = self.output_dir / "rejected_cases" / f"caso_rechazado_{case_id}_{idx:02d}.json"
                if not candidate.exists():
                    path = candidate
                    break
                idx += 1
        self._write_json(path, payload)

    def export_structured_review(self, case, deterministic, audit, error, source_artifact, batch_report, attempt):
        path = self.output_dir / 'human_review_queue' / f'{case.case_id}_structured_review.json'
        if path.exists():
            raise FileExistsError(f'La revisión estructurada ya existe: {path}')
        self._write_json(path, {
            'case_id': case.case_id, 'kind': 'structured', 'status': 'pending',
            'case': case.model_dump(), 'deterministic_validation': deterministic.model_dump(),
            'verification': audit, 'error': error, 'source_artifact': source_artifact,
            'batch_report': Path(batch_report).name, 'attempt': attempt,
            'reviewer_decision': None, 'reviewer_feedback': None,
            'final_case_exported': False,
        })
        return path

    def export_incomplete_review(self, case, deterministic, error, raw, source_artifact, batch_report, attempt):
        path = self.output_dir / 'technical_review_queue' / f'{case.case_id}_incomplete_review.json'
        if path.exists():
            raise FileExistsError(f'La revisión técnica ya existe: {path}')
        self._write_json(path, {
            'case_id': case.case_id, 'status': 'incomplete_review', 'kind': 'technical',
            'case': case.model_dump(), 'deterministic_validation': deterministic.model_dump(),
            'error': error, 'raw_response': raw, 'source_artifact': source_artifact,
            'batch_report': Path(batch_report).name, 'attempt': attempt,
            'clinical_verdict': None, 'automatic_retry': False,
        })
        return path

    def export_artifact(self, case_id: str, artifact_name: str, payload: dict[str, Any]) -> None:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in artifact_name)
        self._write_json(self.output_dir / "generated_artifacts" / f"{case_id}_{safe_name}.json", payload)

    def append_index(self, metadata: CaseMetadata) -> None:
        row = metadata.model_dump()
        exists = self.index_path.exists()
        with self.index_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        write_json(path, data)


def build_metadata(
    case: SyntheticCase,
    deterministic: ValidationResult,
    kimi: KimiReviewResult | None,
    judge: JudgeResult | None,
) -> CaseMetadata:
    events = case.progression.trajectory
    event_types = {e.event_type for e in events}
    return CaseMetadata(
        case_id=case.case_id,
        archetype=case.archetype,
        risk_group=case.risk_classification.risk_group,
        initial_treatment=case.treatment_plan.initial_treatment,
        has_recurrence=any("recurrence" in e for e in event_types),
        has_metastasis=case.diagnosis.tnm.cM.startswith("cM1") or case.diagnosis.metastatic_sites not in {"none", "pelvic_nodes"},
        has_crpc=bool(event_types.intersection({"castration_resistant_progression", "nmCRPC", "mCRPC"})),
        has_palliative_care="palliative_care" in event_types,
        n_events=len(events),
        deterministic_validation_passed=deterministic.passed,
        kimi_review_passed=kimi.passed if kimi else None,
        judge_mean_score=judge.mean_score if judge else None,
        judge_critical_error=judge.critical_error if judge else None,
        judge_clinically_acceptable=judge.clinically_acceptable if judge else None,
        judge_requires_revision=judge.requires_revision if judge else None,
    )
