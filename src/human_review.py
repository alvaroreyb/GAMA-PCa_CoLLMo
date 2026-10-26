from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .batch_orchestrator_agent import BatchOrchestratorAgent
from .config_loader import PROJECT_ROOT, output_directory
from .characterize_batch import characterize, write_json
from .models import SyntheticCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resuelve una solicitud de revisión humana sin modificar el caso estructurado."
    )
    parser.add_argument("case_id", help="Identificador, por ejemplo caso_sintetico_0001")
    parser.add_argument("--decision", choices=("accept", "revise"), required=True)
    parser.add_argument("--feedback", default="", help="Comentario obligatorio para --decision revise")
    return parser.parse_args()


def resolve_review(
    case_id: str,
    decision: str,
    feedback: str = "",
    root: Path = PROJECT_ROOT,
) -> Path:
    data_dir = output_directory(root)
    request_path = data_dir / "human_review_queue" / f"{case_id}_review.json"
    if not request_path.exists():
        raise FileNotFoundError(f"No existe una revisión pendiente para {case_id}")

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload["reviewer_decision"] = decision
    payload["reviewer_feedback"] = feedback
    payload["reviewed_at"] = datetime.now(timezone.utc).isoformat()

    if decision == "accept":
        if payload.get("candidate") and not payload["candidate"]["passes_acceptance"]:
            raise ValueError("La corrección no supera los criterios automáticos; solicita otra corrección.")
        payload["status"] = "accepted_by_human"
        write_json(request_path, payload)
        characterize(data_dir)
        return request_path

    if decision != "revise":
        raise ValueError(f"Decisión desconocida: {decision}")
    if not feedback.strip():
        raise ValueError("--feedback es obligatorio cuando la decisión es revise")

    case_path = data_dir / "generated_json" / f"{case_id}.json"
    narrative_path = data_dir / "generated_txt" / f"{case_id}.txt"
    case = SyntheticCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    original = payload.get("candidate", {}).get("narrative") or narrative_path.read_text(encoding="utf-8")
    orchestrator = BatchOrchestratorAgent(root=root, use_llm_generation=True)
    revised, revision_prompt = orchestrator.narrative_agent.revise(case, original, feedback)
    judge = orchestrator.judge_agent.run(case, revised, enabled=True)

    rejection = orchestrator._narrative_rejection(case, judge, revised) if judge else {"stage": "missing_judge"}
    history = payload.setdefault("revision_history", [])
    candidate = {"narrative": revised, "prompt": revision_prompt,
                 "judge": judge.model_dump() if judge else None,
                 "passes_acceptance": rejection is None,
                 "rejection": rejection, "feedback": feedback,
                 "reviewed_at": payload["reviewed_at"], "version": len(history) + 1}
    version_path = data_dir / "human_revisions" / case_id / f"revision_{candidate['version']:03d}.json"
    write_json(version_path, candidate)
    history.append(str(version_path.relative_to(data_dir)))
    payload.update({"status": "pending_revision" if rejection is None else "revision_failed",
                    "candidate": candidate})
    # Original cohort, metadata and index are immutable; corrected versions carry their own evaluation.
    write_json(request_path, payload)
    characterize(data_dir)
    return request_path


def main() -> None:
    args = parse_args()
    path = resolve_review(args.case_id, args.decision, args.feedback)
    print(path)


if __name__ == "__main__":
    main()
