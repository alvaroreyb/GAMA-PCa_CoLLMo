from __future__ import annotations

import json
import fcntl
import random
import uuid
from pathlib import Path

from .config_loader import PROJECT_ROOT, load_app_config, load_dataset_plan, load_ollama_config, load_yaml, output_directory
from .characterize_batch import characterize, write_json
from .deterministic_validator_agent import DeterministicValidatorAgent
from .deterministic_feedback import feedback_by_owner, repair_scope
from .diagnostic_agent import DiagnosticAgent
from .export_agent import ExportAgent, build_metadata
from .judge_agent import JudgeAgent
from .kimi_case_review_agent import KimiCaseReviewAgent
from .review_candidate_verifier import PendingReviewCandidatesError
from .knowledge_base import KnowledgeBase
from .models import KimiReviewResult, SyntheticCase
from .narrative_generation_agent import NarrativeGenerationAgent
from .ollama_client import OllamaClient, TruncatedResponseError
from .patient_sampler_agent import PatientSamplerAgent
from .progression_agent import ProgressionAgent
from .risk_classification_agent import RiskClassificationAgent
from .treatment_selection_agent import TreatmentSelectionAgent, TreatmentValidationError


class RepeatedValidationError(RuntimeError):
    pass


class KimiReviewUnavailableError(RuntimeError):
    """Orchestration failure, not a change to the historical review agent."""


class KimiReviewPendingError(KimiReviewUnavailableError):
    """A case-level unresolved finding, not a transport or parsing failure."""


class KimiReviewIncompleteError(KimiReviewUnavailableError):
    """Paid but incomplete/unparseable output: preserve it and move to next ID."""
    def __init__(self, message, raw_response=''):
        super().__init__(message)
        self.raw_response = raw_response


class ArchetypeGateError(ValueError):
    def __init__(self, errors: list[str], baseline=None):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.baseline = baseline


class BatchOrchestratorAgent:
    def __init__(
        self,
        root: Path = PROJECT_ROOT,
        use_llm_generation: bool | None = None,
        start_index: int | None = None,
        pipeline_mode: str | None = None,
        early_archetype_gate: bool | None = None,
        dataset_plan_path: Path | None = None,
        knowledge_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        self.root = root
        self.app_config = load_app_config(root)
        self.optimizations = self.app_config.setdefault("optimizations", {})
        for key, value in (("early_archetype_gate", early_archetype_gate),):
            if value is not None:
                self.optimizations[key] = value
        self.ollama_config = load_ollama_config(root)
        self.dataset_plan = load_yaml(dataset_plan_path) if dataset_plan_path is not None else load_dataset_plan(root)
        self.start_index = int(start_index if start_index is not None else self.dataset_plan.get("start_index", 1))
        if use_llm_generation is not None:
            self.app_config.setdefault("app", {})["use_llm_generation"] = use_llm_generation
        if pipeline_mode not in (None, "staged_agents"):
            raise ValueError("Only staged_agents is supported")
        self.kb = KnowledgeBase(root, knowledge_dir=knowledge_dir)
        self.client = OllamaClient.from_config(self.ollama_config)
        logging_cfg = self.app_config.get("logging", {})
        self.client.log_llm_calls = bool(logging_cfg.get("log_llm_calls", True))
        self.client.log_request_messages = bool(logging_cfg.get("log_request_messages", True))
        destination = Path(output_dir).resolve() if output_dir else output_directory(root)
        self.client.trace_dir = destination / "logs" / "llm_calls"
        self.exporter = ExportAgent(destination)
        self.rng = random.Random(int(self.dataset_plan.get("seed", 42)))
        self._init_agents()

    def _init_agents(self) -> None:
        oc = self.ollama_config["ollama"]
        ext = self.ollama_config.get("external_prompts", {})
        self.patient_agent = PatientSamplerAgent("patient_sampler", oc["generation_model"], self.client, "prompts/patient_generation_prompt_es.txt", oc["temperature_generation"], oc["num_predict_generation"])
        self.diagnostic_agent = DiagnosticAgent("diagnostic", oc["generation_model"], self.client, "prompts/diagnosis_generation_prompt_es.txt", oc["temperature_generation"], oc["num_predict_generation"])
        self.risk_agent = RiskClassificationAgent()
        self.treatment_agent = TreatmentSelectionAgent("treatment_selection", oc["generation_model"], self.client, "prompts/treatment_generation_prompt_es.txt", oc["temperature_generation"], oc["num_predict_generation"])
        self.progression_agent = ProgressionAgent("progression", oc["generation_model"], self.client, "prompts/progression_generation_prompt_es.txt", oc["temperature_generation"], oc["num_predict_generation"])
        self.validator = DeterministicValidatorAgent(self.kb)
        self.kimi_agent = KimiCaseReviewAgent("kimi_case_review", oc["validation_model"], self.client, None, oc["temperature_validation"], oc["num_predict_validation"])
        self.kimi_agent.concise_review = bool(self.app_config.get("optimizations", {}).get("kimi_concise_review", False))
        self.kimi_agent.concise_num_predict = int(self.app_config.get("optimizations", {}).get("kimi_concise_num_predict", 1600))
        self.kimi_agent.candidate_review = bool(self.optimizations.get("kimi_candidate_review", False))
        if self.kimi_agent.concise_num_predict <= 0:
            raise ValueError("kimi_concise_num_predict must be positive")
        self.review_provider = "ollama"
        self.kimi_agent.cache_friendly_prefix = True
        self.narrative_agent = NarrativeGenerationAgent("narrative_generation", oc["narrative_model"], self.client, "prompts/narrative_realistic_generation_prompt_es.txt", oc["temperature_narrative"], oc["num_predict_narrative"], external_prompt_path=ext.get("narrative_prompt_path"))
        self.judge_agent = JudgeAgent(
            "judge",
            oc["judge_model"],
            self.client,
            "prompts/clinical_rubric_judge_parametric_prompt_es.txt",
            oc["temperature_judge"],
            oc["num_predict_judge"],
            external_prompt_path=ext.get("judge_prompt_path"),
        )

    def run(self, total_cases: int | None = None, resume: bool = False,
            max_new_positions: int | None = None, pending_only: bool = False) -> list[SyntheticCase]:
        if pending_only and not resume:
            raise ValueError('pending_only requires resume')
        if max_new_positions is not None and max_new_positions < 1:
            raise ValueError("max_new_positions must be positive")
        with (self.exporter.output_dir / ".generation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Ya existe una generación activa en esta carpeta de salida") from None
            return self._run(total_cases, resume, max_new_positions, pending_only)

    def _run(self, total_cases: int | None = None, resume: bool = False,
             max_new_positions: int | None = None, pending_only: bool = False) -> list[SyntheticCase]:
        self.kimi_agent.reuse_logged_response = resume
        plan_total = int(total_cases if total_cases is not None else self.dataset_plan.get("total_cases", 100))
        generated: list[SyntheticCase] = []
        counters = {k: 0 for k in self.dataset_plan["quotas_by_archetype"]}
        sequence = self._archetype_sequence(plan_total)
        if plan_total <= 0 or len(sequence) != plan_total:
            raise ValueError("Requested case count must be positive and must not exceed the configured quotas")
        case_ids = [f"caso_sintetico_{self.start_index + i:04d}" for i in range(plan_total)]
        self.batch_report_path = self.exporter.output_dir / f"batch_{self.start_index:04d}_{self.start_index + plan_total - 1:04d}.json"
        previous = json.loads(self.batch_report_path.read_text()) if self.batch_report_path.exists() else None
        outcomes = list(previous.get("outcomes", [])) if resume and previous else []
        if resume and previous:
            if previous.get("planned") != plan_total or len(outcomes) != previous.get("processed"):
                raise ValueError("El informe previo no coincide con el plan solicitado")
            for index, item in enumerate(outcomes):
                if item["case_id"] != case_ids[index] or item["archetype"] != sequence[index]:
                    raise ValueError("El orden o las cuotas del lote han cambiado; no se puede reanudar")
                if item["status"] == "accepted":
                    case = SyntheticCase.model_validate_json((self.exporter.output_dir / "generated_json" / f"{item['case_id']}.json").read_text())
                    generated.append(case)
                    counters[item["archetype"]] += 1
        for cid in case_ids[len(outcomes):]:
            for subdir in ("generated_json", "generated_txt", "generated_metadata", "human_review_queue"):
                if any((self.exporter.output_dir / subdir).glob(f"{cid}*")):
                    raise FileExistsError(f"Existing output for {cid}; choose an unused ID range")
        if self.batch_report_path.exists() and not resume:
            previous = json.loads(self.batch_report_path.read_text(encoding="utf-8"))
            empty = (previous.get("status") == "running"
                     and previous.get("planned") == plan_total
                     and previous.get("processed") == 0
                     and previous.get("accepted") == 0
                     and previous.get("rejected") == 0
                     and previous.get("outcomes") == [])
            has_artifacts = any(
                any((self.exporter.output_dir / "generated_artifacts").glob(f"{cid}_*.json"))
                for cid in case_ids
            )
            if not empty or has_artifacts:
                raise FileExistsError(f"Existing batch report with results or artifacts: {self.batch_report_path}")
            archive = self.exporter.output_dir / "batch_history" / f"{self.batch_report_path.stem}_{uuid.uuid4().hex}.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            self.batch_report_path.rename(archive)
            print(f"Restarting empty batch; previous report preserved at {archive}", flush=True)
        sessions = list(previous.get("sessions", [])) if previous else []
        sessions.append({"run_id": uuid.uuid4().hex, "resume": resume,
                         "starting_position": len(outcomes), "ollama": self.ollama_config["ollama"],
                         "optimizations": dict(self.optimizations), "max_new_positions": max_new_positions,
                         "kimi_review_provider": self.review_provider,
                         "kimi_candidate_review": self.kimi_agent.candidate_review,
                         "kimi_review_model": self.kimi_agent.model_name,
                         "kimi_review_cache_friendly_prefix": self.kimi_agent.cache_friendly_prefix,
                         "kimi_review_mode": "concise_local_parse" if self.kimi_agent.concise_review else "candidate_verification",
                         "kimi_review_output_limit": min(self.kimi_agent.num_predict, self.kimi_agent.concise_num_predict) if self.kimi_agent.concise_review else self.kimi_agent.num_predict})
        run_prefix = f"resume_{sessions[-1]['run_id'][:12]}_" if resume else ""
        starting_processed = len(outcomes)

        def save_report(status):
            write_json(self.batch_report_path, {
                "status": status,
                "planned": plan_total,
                "processed": len(outcomes),
                "accepted": len(generated),
                "rejected": sum(item["status"] == "rejected" for item in outcomes),
                "pending_review": sum(item["status"] == "pending_review" for item in outcomes),
                "awaiting_narrative": sum(item["status"] == "ready_for_narrative" for item in outcomes),
                "incomplete_review": sum(item["status"] == "incomplete_review" for item in outcomes),
                "accepted_by_archetype": counters,
                "outcomes": outcomes,
                "sessions": sessions,
            })
            characterize(self.exporter.output_dir, self.start_index, self.start_index + plan_total - 1)

        save_report("running")
        max_attempts = int(self.dataset_plan.get("max_attempts_per_case", 10))
        max_narrative_attempts = int(self.dataset_plan.get("max_narrative_attempts_per_case", 3))
        mode = self.app_config.get("app", {}).get("pipeline_mode", "staged_agents")
        use_llm = bool(self.app_config.get("app", {}).get("use_llm_generation", True))
        print(f"Generating {plan_total} cases from index {self.start_index:04d} with BatchOrchestratorAgent mode={mode} use_llm={use_llm}", flush=True)
        for idx, archetype in enumerate(sequence, start=1):
            recovering = idx <= len(outcomes) and outcomes[idx - 1]['status'] == 'ready_for_narrative'
            if pending_only and not recovering:
                continue
            if idx <= len(outcomes) and not recovering:
                continue
            if max_new_positions is not None and len(outcomes) - starting_processed >= max_new_positions:
                save_report("paused_pilot_limit")
                return generated
            case_id = case_ids[idx - 1]
            recovered_payload = None
            independent_technical_review = False
            if recovering:
                recovery_queue = outcomes[idx - 1].get('recovery_queue', f'human_review_queue/{case_id}_structured_review.json')
                relative_queue = Path(recovery_queue)
                if (len(relative_queue.parts) != 2 or relative_queue.parts[0] not in ('human_review_queue', 'technical_review_queue')
                        or not relative_queue.name.startswith(f'{case_id}_')):
                    raise ValueError('Ruta de recuperación inválida')
                recovered_path = self.exporter.output_dir / relative_queue
                recovered_payload = json.loads(recovered_path.read_text())
                independent_technical_review = recovered_payload.get('kind') == 'technical'
                if recovered_payload.get('status') != 'ready_for_narrative' or not recovered_payload.get('automatic_review', {}).get('passed'):
                    raise ValueError(f'{case_id}: checkpoint de recuperación incoherente')
            accepted = False
            pending_review = False
            incomplete_review = False
            self._validation_feedback = []
            self._treatment_retry = None
            progression_retry = None
            retry_source = None
            if resume and mode == "staged_agents":
                restored = self._load_progression_retry(case_id, archetype)
                if restored:
                    progression_retry, self._validation_feedback, retry_source = restored
            last_errors = None
            repeated_errors = 0
            restart_markers = list((self.exporter.output_dir / 'generated_artifacts').glob(f'{case_id}_*00_full_restart.json'))
            full_restart_used = bool(restart_markers)
            # Resume an old repeated-error pause directly with the fresh attempt,
            # rather than paying for the same partial repairs three more times.
            full_restart_pending = bool(resume and previous and
                                        previous.get('status') == 'paused_repeated_validation_errors'
                                        and idx == starting_processed + 1 and not full_restart_used
                                        and (progression_retry is None or bool(self._validation_feedback)))
            restart_exhausted = full_restart_used and progression_retry is not None and bool(self._validation_feedback)
            failure_reason = 'full_restart_validation_failed' if restart_exhausted else 'structured_attempts_exhausted'
            attempt = 0
            while attempt < max_attempts or full_restart_pending:
                if restart_exhausted:
                    break
                attempt += 1
                self._artifact_prefix = f"{run_prefix}attempt_{attempt:02d}_"
                print(f"[{idx}/{plan_total}] {case_id} archetype={archetype} attempt={attempt}", flush=True)
                try:
                    if full_restart_pending:
                        self.exporter.export_artifact(case_id, f'{self._artifact_prefix}00_full_restart', {
                            'case_id': case_id, 'scope': 'full_case_once', 'attempt': attempt,
                            'reason': 'repeated_validation_errors', 'previous_errors': list(self._validation_feedback),
                            'source_artifact': retry_source, 'reused_blocks': [],
                        })
                        progression_retry = None
                        recovered_payload = None
                        self._treatment_retry = None
                        self._validation_feedback = []
                        retry_source = None
                        repeated_errors = 0
                        last_errors = None
                        full_restart_used = True
                        full_restart_pending = False
                        print(f'  full-case restart (once): {case_id}', flush=True)
                    if recovered_payload is not None:
                        case = SyntheticCase.model_validate(recovered_payload['case'])
                        if case.case_id != case_id or case.archetype != archetype:
                            raise ValueError('Identidad del esqueleto recuperado no coincide con el lote')
                        print(f'  reusing CK-filtered skeleton and paid review: {case_id}', flush=True)
                    elif progression_retry is not None:
                        if self._validation_feedback:
                            scope = self._repair_scope(progression_retry, self._validation_feedback)
                            case = (self._regenerate_progression(progression_retry, retry_source)
                                    if scope == "progression" else self._regenerate_blocks(progression_retry, retry_source))
                        else:
                            # A previously deterministic-rejected checkpoint now
                            # passes (e.g. a validator false positive was fixed).
                            case = progression_retry.model_copy(deep=True)
                            self.exporter.export_artifact(case_id, f"{self._artifact_prefix}00_retry_scope", {
                                "case_id": case_id, "scope": "structured_reuse_after_revalidation",
                                "source_artifact": retry_source,
                                "reused_blocks": ["patient", "diagnosis", "risk_classification", "treatment_plan", "progression"],
                                "validation_feedback": [],
                            })
                            print(f"  reusing structured candidate now passing validation: {case_id}", flush=True)
                    else:
                        case = self._build_case(case_id, archetype)
                    self.exporter.export_artifact(
                        case_id,
                        f"{self._artifact_prefix}01_structured_case",
                        {
                            "case_id": case_id,
                            "archetype": archetype,
                            "pipeline_mode": mode,
                            "use_llm_generation": use_llm,
                            "case": case.model_dump(),
                        },
                    )
                    deterministic = self.validator.run(case)
                    self.exporter.export_artifact(
                        case_id,
                        f"{self._artifact_prefix}02_deterministic_validation",
                        {"case_id": case_id, "validation": deterministic.model_dump()},
                    )
                    if not deterministic.passed:
                        self.exporter.export_rejected(case_id, {"stage": "deterministic_validation", "errors": deterministic.errors, "case": case.model_dump()})
                        print(f"  rejected deterministic_validation: {case_id}", flush=True)
                        self._validation_feedback = list(deterministic.errors)
                        retry_source = f"{case_id}_{self._artifact_prefix}01_structured_case.json"
                        progression_retry = case if mode == "staged_agents" and self._repair_scope(case, deterministic.errors) else None
                        signature = tuple(sorted(deterministic.errors))
                        repeated_errors = repeated_errors + 1 if signature == last_errors else 1
                        last_errors = signature
                        limit = int(self.dataset_plan.get("stop_after_repeated_validation_errors", 3))
                        if full_restart_used:
                            failure_reason = 'full_restart_validation_failed'
                            print(f'  full restart failed validation; skipping {case_id}', flush=True)
                            break
                        if limit and repeated_errors >= limit:
                            full_restart_pending = True
                            print(f'  {limit} identical validation failures; scheduling one full restart: {case_id}', flush=True)
                        continue
                    repeated_errors = 0
                    last_errors = None
                    # Downstream clinical-review/narrative failures must not be
                    # mistaken for progression-only deterministic failures.
                    progression_retry = None
                    self._validation_feedback = []
                    print(f"  running kimi case review: {case_id}", flush=True)
                    kimi = (KimiReviewResult.model_validate(recovered_payload['automatic_review'])
                            if recovered_payload is not None else self._run_kimi(case))
                    if kimi:
                        review_stage = '03_independent_ck_review' if independent_technical_review else '03_kimi_case_review'
                        self.exporter.export_artifact(case_id, f"{self._artifact_prefix}{review_stage}", {
                            "case_id": case_id, "review": kimi.model_dump(),
                            'review_source': 'assisted_ck_audit' if independent_technical_review else self.review_provider})
                    if kimi and not kimi.passed and self.dataset_plan.get("reject_if_kimi_review_fails", True):
                        self.exporter.export_rejected(case_id, {"stage": "kimi_review", "review": kimi.model_dump(), "case": case.model_dump()})
                        print(f"  rejected kimi_review: {case_id}", flush=True)
                        if full_restart_used:
                            failure_reason = 'full_restart_kimi_rejected'
                            break
                        continue
                    narrative_result = self._run_narrative_attempts(case, case_id, max_narrative_attempts)
                    if narrative_result is None:
                        print(f"  rejected narrative_attempts_exhausted: {case_id}", flush=True)
                        if full_restart_used:
                            failure_reason = 'full_restart_narrative_failed'
                            break
                        continue
                    narrative, prompt, judge = narrative_result
                    metadata = build_metadata(case, deterministic, kimi, judge)
                    if independent_technical_review:
                        metadata.kimi_review_passed = None
                        case.metadata['technical_review_resolution'] = recovered_payload['assisted_review']
                    self.exporter.export_case(case, narrative, prompt, metadata,
                                              None if independent_technical_review else kimi, judge)
                    generated.append(case)
                    counters[archetype] += 1
                    accepted = True
                    print(f"  accepted {case_id}; generated={len(generated)}/{plan_total}", flush=True)
                    break
                except KimiReviewPendingError as exc:
                    audit = self.kimi_agent.last_candidate_review
                    queue_path = self.exporter.export_structured_review(
                        case, deterministic, audit, str(exc),
                        f"{case_id}_{self._artifact_prefix}01_structured_case.json",
                        self.batch_report_path, attempt)
                    pending_review = True
                    print(f"  queued structured_review: {case_id}; {queue_path}", flush=True)
                    break
                except KimiReviewIncompleteError as exc:
                    self.exporter.export_incomplete_review(
                        case, deterministic, str(exc), exc.raw_response,
                        f'{case_id}_{self._artifact_prefix}01_structured_case.json',
                        self.batch_report_path, attempt)
                    incomplete_review = True
                    print(f'  incomplete kimi review preserved; skipping to next ID: {case_id}', flush=True)
                    break
                except KimiReviewUnavailableError as exc:
                    self.exporter.export_artifact(case_id, f"{self._artifact_prefix}03_kimi_review_unavailable", {
                        "case_id": case_id, "stage": "kimi_review_unavailable", "error": str(exc),
                        "source_artifact": f"{case_id}_{self._artifact_prefix}01_structured_case.json",
                    })
                    save_report("paused_kimi_review_unavailable")
                    # Propagate past the generic candidate-exception handler:
                    # a missing review must never regenerate a valid case.
                    raise
                except RepeatedValidationError:
                    raise
                except ArchetypeGateError as exc:
                    self.exporter.export_rejected(case_id, {"stage": "early_archetype_gate", "archetype": archetype,
                                                           "attempt": attempt, "errors": exc.errors})
                    self._validation_feedback = list(exc.errors)
                    if exc.baseline is not None:
                        self._treatment_retry = exc.baseline
                    signature = tuple(sorted(exc.errors))
                    repeated_errors = repeated_errors + 1 if signature == last_errors else 1
                    last_errors = signature
                    limit = int(self.dataset_plan.get("stop_after_repeated_validation_errors", 3))
                    print(f"  rejected early_archetype_gate: {case_id}: {exc}", flush=True)
                    if full_restart_used:
                        failure_reason = 'full_restart_archetype_failed'
                        break
                    if limit and repeated_errors >= limit:
                        full_restart_pending = True
                        print(f'  {limit} identical archetype failures; scheduling one full restart: {case_id}', flush=True)
                except Exception as exc:
                    self.exporter.export_rejected(case_id, {"stage": "exception", "archetype": archetype, "attempt": attempt, "error": str(exc)})
                    print(f"  rejected exception: {case_id}: {exc}", flush=True)
                    if full_restart_used:
                        failure_reason = 'full_restart_exception'
                        break
            if not accepted and not pending_review and not incomplete_review:
                self.exporter.export_rejected(case_id, {
                    "stage": failure_reason,
                    "archetype": archetype,
                    "max_attempts": max_attempts,
                })
            outcome = {"case_id": case_id, "archetype": archetype,
                       "status": "accepted" if accepted else "incomplete_review" if incomplete_review else "pending_review" if pending_review else "rejected",
                       "attempts": outcomes[idx - 1]['attempts'] if recovering else attempt}
            if full_restart_used:
                outcome['full_restart_used'] = True
            if not accepted and not pending_review and not incomplete_review:
                outcome['rejection_reason'] = failure_reason
            if recovering:
                outcome['review_resolution'] = 'assisted_ck_audit' if independent_technical_review else 'automatic_ck_filter'
                if 'recovery_queue' in outcomes[idx - 1]:
                    outcome['recovery_queue'] = outcomes[idx - 1]['recovery_queue']
                outcomes[idx - 1] = outcome
                recovered_payload.update(status='resolved_automatically' if accepted else 'incomplete_review' if incomplete_review else 'downstream_rejected',
                                         final_case_exported=accepted)
                write_json(recovered_path, recovered_payload)
            else:
                outcomes.append(outcome)
            save_report("running")
        save_report("paused_pending_only" if pending_only else "completed")
        return generated

    def _can_retry_progression(self, case, validation) -> bool:
        if not self.validator.progression_only_failure(validation):
            return False
        return self._baseline_risk_compatible(case)

    def _repair_scope(self, case, errors):
        scope = repair_scope(errors)
        if scope in {"progression", "treatment"} and not self._baseline_risk_compatible(case):
            return None
        if scope == "risk":
            try:
                risk = self.risk_agent.run(case.diagnosis)
            except ValueError:
                return None
            if risk.risk_group not in self.kb.archetype(case.archetype).get("target_risk_groups", []):
                return None
        return scope

    def _baseline_risk_compatible(self, case) -> bool:
        # Recompute classification before freezing baseline blocks, including
        # when loading a checkpoint made with an earlier version/configuration.
        try:
            classified = self.risk_agent.run(case.diagnosis)
        except ValueError:
            return False
        return (classified.risk_group == case.risk_classification.risk_group
                and classified.risk_group in self.kb.archetype(case.archetype).get("target_risk_groups", []))

    def _load_progression_retry(self, case_id, archetype):
        """Reuse only the newest structured candidate, revalidated under today's rules."""
        paths = list((self.exporter.output_dir / "generated_artifacts").glob(f"{case_id}_*01_structured_case.json"))
        if not paths:
            return None
        path = max(paths, key=lambda p: (p.stat().st_mtime_ns, p.name))
        try:
            case = SyntheticCase.model_validate(json.loads(path.read_text())["case"])
            if case.case_id != case_id or case.archetype != archetype or case.metadata.get("pipeline_mode") != "staged_agents":
                return None
            validation = self.validator.run(case)
            if not validation.passed and self._repair_scope(case, validation.errors):
                return case, list(validation.errors), path.name
            recorded_path = path.with_name(path.name.replace("01_structured_case.json", "02_deterministic_validation.json"))
            if validation.passed and self._baseline_risk_compatible(case) and recorded_path.exists():
                recorded = json.loads(recorded_path.read_text())["validation"]
                # No reuse of an already valid candidate rejected downstream:
                # only one originally stopped at deterministic validation.
                if recorded.get("passed") is False:
                    return case, [], path.name
                prefix = path.name.removesuffix("01_structured_case.json")
                artifact_dir = path.parent
                unfinished_review = not (artifact_dir / f"{prefix}03_kimi_case_review.json").exists()
                has_downstream = any(artifact_dir.glob(f"{prefix}*04_narrative_generation.json")) or any(artifact_dir.glob(f"{prefix}*05_judge.json"))
                if recorded.get("passed") is True and unfinished_review and not has_downstream:
                    return case, [], path.name
        except (ValueError, KeyError, OSError):
            # Never overwrite or silently repair a corrupt checkpoint.
            return None
        return None

    def _regenerate_blocks(self, case: SyntheticCase, source: str | None) -> SyntheticCase:
        errors = list(self._validation_feedback)
        scope = self._repair_scope(case, errors)
        if scope is None:
            raise ValueError("No safe deterministic feedback route")
        grouped = feedback_by_owner(errors)
        updated = case.model_copy(deep=True)
        use_llm = bool(self.app_config.get("app", {}).get("use_llm_generation", True))
        changed = []

        def apply_block(field, callback):
            frozen = updated.model_dump(exclude={field})
            value = callback()
            if updated.model_dump(exclude={field}) != frozen:
                raise ValueError(f"{field} repair modified frozen input blocks")
            setattr(updated, field, value)
            changed.append(field)
            stage = {"patient": "00_patient", "diagnosis": "00_diagnosis",
                     "risk_classification": "00_risk", "treatment_plan": "00_treatment",
                     "progression": "00_progression"}[field]
            self.exporter.export_artifact(case.case_id, f"{self._artifact_prefix}{stage}", {
                "case_id": case.case_id, field: value.model_dump(),
            })

        print(f"  deterministic correction: {case.case_id}; owner={scope}", flush=True)
        self.exporter.export_artifact(case.case_id, f"{self._artifact_prefix}00_retry_scope", {
            "case_id": case.case_id, "scope": scope, "source_artifact": source,
            "feedback_by_agent": grouped, "validation_feedback": errors,
        })
        if scope == "patient":
            apply_block("patient", lambda: self.patient_agent.run(
                updated.archetype, self.kb, self.dataset_plan, use_llm=use_llm, rng=self.rng,
                validation_feedback=grouped["patient"], previous_patient=case.patient.model_copy(deep=True)))
            invalid_fields = {error.split(":", 2)[1].strip().split(".")[0] for error in grouped["patient"]}
            if updated.patient.model_dump(exclude=invalid_fields) != case.patient.model_dump(exclude=invalid_fields):
                raise ValueError("Patient repair modified valid fields or biomarkers")
        if scope == "patient" or grouped["diagnosis"]:
            apply_block("diagnosis", lambda: self.diagnostic_agent.run(
                updated.patient, updated.archetype, self.kb, use_llm=use_llm, rng=self.rng,
                validation_feedback=grouped["diagnosis"], previous_diagnosis=case.diagnosis.model_copy(deep=True)))
            if grouped["diagnosis"] and all(error.split(":", 1)[0] == "psa_density_consistency" for error in grouped["diagnosis"]):
                if updated.diagnosis.model_dump(exclude={"psa_density"}) != case.diagnosis.model_dump(exclude={"psa_density"}):
                    raise ValueError("PSA-density correction modified unrelated diagnostic fields")
        if scope in {"patient", "diagnosis", "risk"}:
            apply_block("risk_classification", lambda: self.risk_agent.run(updated.diagnosis))
        clinical_diagnosis_changed = (updated.diagnosis.model_dump(exclude={"psa_density"})
                                      != case.diagnosis.model_dump(exclude={"psa_density"}))
        upstream_changed = (updated.patient != case.patient or clinical_diagnosis_changed
                            or updated.risk_classification.risk_group != case.risk_classification.risk_group)
        if grouped["treatment"] or upstream_changed:
            apply_block("treatment_plan", lambda: self.treatment_agent.run(
                updated.patient, updated.diagnosis, updated.risk_classification, updated.archetype,
                self.kb, use_llm=use_llm, validation_feedback=grouped["treatment"],
                previous_treatment=case.treatment_plan.model_copy(deep=True)))
        if grouped["progression"] or upstream_changed or updated.treatment_plan != case.treatment_plan:
            apply_block("progression", lambda: self.progression_agent.run(
                updated.patient, updated.diagnosis, updated.risk_classification, updated.treatment_plan,
                updated.archetype, self.kb, use_llm=use_llm, rng=self.rng,
                validation_feedback=grouped["progression"],
                previous_progression=case.progression.model_copy(deep=True)))
        fields = ["patient", "diagnosis", "risk_classification", "treatment_plan", "progression"]
        self.exporter.export_artifact(case.case_id, f"{self._artifact_prefix}00_retry_scope", {
            "case_id": case.case_id, "scope": scope, "source_artifact": source,
            "feedback_by_agent": grouped, "validation_feedback": errors,
            "regenerated_blocks": changed, "reused_blocks": [field for field in fields if field not in changed],
        })
        # Acceptance remains the responsibility of the next FULL validation,
        # followed by the unchanged Kimi review and narrative judge.
        return updated

    def _regenerate_progression(self, case: SyntheticCase, source: str | None) -> SyntheticCase:
        print(f"  progression-only retry: {case.case_id}; reusing patient/diagnosis/risk/treatment", flush=True)
        self.exporter.export_artifact(case.case_id, f"{self._artifact_prefix}00_retry_scope", {
            "case_id": case.case_id, "scope": "progression_only", "source_artifact": source,
            "reused_blocks": ["patient", "diagnosis", "risk_classification", "treatment_plan"],
            "validation_feedback": list(self._validation_feedback),
        })
        # Deep copy protects the previous candidate even if an agent mutates inputs.
        updated = case.model_copy(deep=True)
        updated.progression = self.progression_agent.run(
            updated.patient, updated.diagnosis, updated.risk_classification, updated.treatment_plan,
            updated.archetype, self.kb,
            use_llm=bool(self.app_config.get("app", {}).get("use_llm_generation", True)), rng=self.rng,
            validation_feedback=self._validation_feedback, previous_progression=case.progression.model_copy(deep=True),
        )
        if updated.model_dump(exclude={"progression"}) != case.model_dump(exclude={"progression"}):
            raise ValueError("Progression-only retry modified frozen baseline blocks")
        self.exporter.export_artifact(case.case_id, f"{self._artifact_prefix}00_progression", {
            "case_id": case.case_id, "progression": updated.progression.model_dump(),
        })
        return updated

    def _run_narrative_attempts(self, case: SyntheticCase, case_id: str, max_attempts: int):
        last_payload = None
        for narrative_attempt in range(1, max_attempts + 1):
            artifact_prefix = self._artifact_prefix
            print(f"  generating narrative: {case_id} narrative_attempt={narrative_attempt}/{max_attempts}", flush=True)
            narrative, prompt = self._run_narrative(case)
            self.exporter.export_artifact(
                case_id,
                f"{artifact_prefix}narrative_{narrative_attempt:02d}_04_narrative_generation",
                {
                    "case_id": case_id,
                    "narrative_attempt": narrative_attempt,
                    "prompt": prompt,
                    "narrative": narrative,
                },
            )
            print(f"  judging narrative: {case_id} narrative_attempt={narrative_attempt}/{max_attempts}", flush=True)
            judge = self._run_judge(case, narrative)
            if judge:
                self.exporter.export_artifact(
                    case_id,
                    f"{artifact_prefix}narrative_{narrative_attempt:02d}_05_judge",
                    {"case_id": case_id, "narrative_attempt": narrative_attempt, "judge": judge.model_dump()},
                )

            rejection = self._narrative_rejection(case, judge, narrative)
            if rejection is None:
                return narrative, prompt, judge

            last_payload = {
                "stage": rejection["stage"],
                "narrative_attempt": narrative_attempt,
                "max_narrative_attempts": max_attempts,
                "case": case.model_dump(),
                "narrative": narrative,
                **rejection,
            }
            self.exporter.export_rejected(case_id, last_payload)
            detail = f" mean={rejection['mean']}" if "mean" in rejection else ""
            print(f"  rejected {rejection['stage']}: {case_id} narrative_attempt={narrative_attempt}/{max_attempts}{detail}", flush=True)

        if last_payload:
            self.exporter.export_rejected(
                case_id,
                {
                    "stage": "narrative_attempts_exhausted",
                    "max_narrative_attempts": max_attempts,
                    "last_rejection": last_payload,
                    "case": case.model_dump(),
                },
            )
        return None

    def _narrative_rejection(self, case: SyntheticCase, judge, narrative: str) -> dict | None:
        if judge and self.dataset_plan.get("reject_if_judge_critical_error", True) and judge.critical_error:
            return {"stage": "judge_critical_error", "judge": judge.model_dump()}
        min_mean = float(self.dataset_plan.get("judge_minimum_mean_score", 4.5))
        if judge and min_mean:
            judge_mean = self._judge_mean(judge)
            if judge_mean is None or judge_mean < min_mean:
                return {
                    "stage": "judge_mean_below_threshold",
                    "threshold": min_mean,
                    "mean": judge_mean,
                    "judge": judge.model_dump(),
                }
        return None

    @staticmethod
    def _judge_mean(judge) -> float | None:
        scores = [item.score for item in judge.scores.values()]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _build_case(self, case_id: str, archetype: str) -> SyntheticCase:
        use_llm = bool(self.app_config.get("app", {}).get("use_llm_generation", True))
        pending = getattr(self, "_treatment_retry", None)
        if pending is not None and pending["case_id"] == case_id and pending["archetype"] == archetype:
            return self._finish_treatment_retry(pending)
        print(f"  patient_sampler agent: {case_id}", flush=True)
        patient = self.patient_agent.run(archetype, self.kb, self.dataset_plan, use_llm=use_llm, rng=self.rng)
        prefix = getattr(self, "_artifact_prefix", "")
        self.exporter.export_artifact(case_id, f"{prefix}00_patient", {"case_id": case_id, "archetype": archetype, "patient": patient.model_dump()})
        print(f"  diagnostic agent: {case_id}", flush=True)
        diagnosis = self.diagnostic_agent.run(patient, archetype, self.kb, use_llm=use_llm, rng=self.rng)
        self.exporter.export_artifact(case_id, f"{prefix}00_diagnosis", {"case_id": case_id, "diagnosis": diagnosis.model_dump()})
        print(f"  risk classifier: {case_id}", flush=True)
        risk = self.risk_agent.run(diagnosis)
        self.exporter.export_artifact(case_id, f"{prefix}00_risk", {"case_id": case_id, "risk": risk.model_dump()})
        target_groups = self.kb.archetype(archetype).get("target_risk_groups", [])
        if risk.risk_group not in target_groups:
            raise ValueError(f"Risk group {risk.risk_group} incompatible with archetype {archetype}")
        print(f"  treatment_selection agent: {case_id}", flush=True)
        baseline = {"case_id": case_id, "archetype": archetype, "patient": patient,
                    "diagnosis": diagnosis, "risk": risk}
        try:
            treatment = self.treatment_agent.run(patient, diagnosis, risk, archetype, self.kb, use_llm=use_llm)
        except TreatmentValidationError as exc:
            raise ArchetypeGateError(exc.errors, {**baseline, "treatment": exc.plan}) from exc
        self.exporter.export_artifact(case_id, f"{prefix}00_treatment", {"case_id": case_id, "treatment": treatment.model_dump()})
        if self.optimizations.get("early_archetype_gate", False):
            errors = self.validator.validate_archetype_baseline(archetype, risk.risk_group, treatment.initial_treatment)
            self.exporter.export_artifact(case_id, f"{prefix}00_archetype_gate", {"passed": not errors, "errors": errors})
            if errors:
                raise ArchetypeGateError(errors, {**baseline, "treatment": treatment})
        print(f"  progression agent: {case_id}", flush=True)
        progression = self.progression_agent.run(patient, diagnosis, risk, treatment, archetype, self.kb, use_llm=use_llm, rng=self.rng,
                                                validation_feedback=getattr(self, "_validation_feedback", None))
        self.exporter.export_artifact(case_id, f"{prefix}00_progression", {"case_id": case_id, "progression": progression.model_dump()})
        return SyntheticCase(case_id=case_id, archetype=archetype, patient=patient, diagnosis=diagnosis, risk_classification=risk, treatment_plan=treatment, progression=progression, metadata={"fictitious": True, "pipeline_mode": self.app_config.get("app", {}).get("pipeline_mode", "staged_agents")})

    def _finish_treatment_retry(self, baseline):
        case_id, archetype = baseline["case_id"], baseline["archetype"]
        patient, diagnosis, risk = (baseline[name].model_copy(deep=True) for name in ("patient", "diagnosis", "risk"))
        frozen = [block.model_dump() for block in (patient, diagnosis, risk)]
        use_llm = bool(self.app_config.get("app", {}).get("use_llm_generation", True))
        self.exporter.export_artifact(case_id, f"{self._artifact_prefix}00_retry_scope", {
            "case_id": case_id, "scope": "treatment_before_progression",
            "reused_blocks": ["patient", "diagnosis", "risk_classification"],
            "validation_feedback": list(self._validation_feedback),
        })
        if baseline.get("treatment_verified"):
            treatment = baseline["treatment"].model_copy(deep=True)
        else:
            try:
                treatment = self.treatment_agent.run(
                    patient, diagnosis, risk, archetype, self.kb, use_llm=use_llm,
                    validation_feedback=self._validation_feedback, previous_treatment=baseline["treatment"].model_copy(deep=True))
            except TreatmentValidationError as exc:
                raise ArchetypeGateError(exc.errors, {**baseline, "treatment": exc.plan}) from exc
        if [block.model_dump() for block in (patient, diagnosis, risk)] != frozen:
            raise ValueError("Treatment repair modified frozen baseline blocks")
        errors = self.validator.validate_archetype_baseline(archetype, risk.risk_group, treatment.initial_treatment)
        if errors:
            raise ArchetypeGateError(errors, {**baseline, "treatment": treatment})
        self.exporter.export_artifact(case_id, f"{self._artifact_prefix}00_treatment", {"case_id": case_id, "treatment": treatment.model_dump()})
        self._treatment_retry = {**baseline, "treatment": treatment.model_copy(deep=True), "treatment_verified": True}
        progression = self.progression_agent.run(patient, diagnosis, risk, treatment, archetype, self.kb, use_llm=use_llm, rng=self.rng)
        self.exporter.export_artifact(case_id, f"{self._artifact_prefix}00_progression", {"case_id": case_id, "progression": progression.model_dump()})
        self._treatment_retry = None
        return SyntheticCase(case_id=case_id, archetype=archetype, patient=patient, diagnosis=diagnosis,
                             risk_classification=risk, treatment_plan=treatment, progression=progression,
                             metadata={"fictitious": True, "pipeline_mode": "staged_agents"})

    def _run_kimi(self, case: SyntheticCase) -> KimiReviewResult | None:
        if not self.dataset_plan.get("use_kimi_case_review", True):
            return None
        if not self.app_config.get("app", {}).get("use_llm_generation", True):
            return KimiReviewResult(passed=True, clinical_plausibility_score=10)
        try:
            return self.kimi_agent.run(case, self.kb, enabled=True)
        except TruncatedResponseError as exc:
            raise KimiReviewIncompleteError(f'{case.case_id}: revisión truncada; no repetir ni regenerar', exc.content) from exc
        except PendingReviewCandidatesError as exc:
            raise KimiReviewPendingError(f"{case.case_id}: {exc}") from exc
        except ValueError as exc:
            # JSON/contract failures are case-level output failures, not clinical rejections.
            raw = getattr(self.kimi_agent, 'last_raw_review', None) or ''
            raise KimiReviewIncompleteError(f'{case.case_id}: contrato de revisión inválido ({exc})', raw) from exc
        except Exception as exc:
            # A transport/parsing/contract failure must not regenerate a valid case.
            raise KimiReviewUnavailableError(
                f"{case.case_id}: revisión no disponible ({type(exc).__name__}: {exc})"
            ) from exc
        finally:
            audit = getattr(self.kimi_agent, 'last_candidate_review', None)
            if self.kimi_agent.candidate_review and isinstance(audit, dict):
                self.exporter.export_artifact(case.case_id, f"{getattr(self, '_artifact_prefix', '')}03_candidate_verification",
                                             {'case_id': case.case_id, 'verification': audit})

    def _run_narrative(self, case: SyntheticCase) -> tuple[str, str]:
        enabled = bool(self.dataset_plan.get("generate_narrative", True))
        if not self.app_config.get("app", {}).get("use_llm_generation", True):
            prompt = self.narrative_agent.build_prompt(case)
            text = "" if not enabled else self._fallback_narrative(case)
            return text, prompt
        return self.narrative_agent.run(case, enabled=enabled)

    def _run_judge(self, case: SyntheticCase, narrative: str):
        if not self.dataset_plan.get("use_judge", True) or not narrative:
            return None
        if not self.app_config.get("app", {}).get("use_llm_generation", True):
            return None
        return self.judge_agent.run(case, narrative, enabled=True)

    def _archetype_sequence(self, total: int) -> list[str]:
        remaining = {k: int(v) for k, v in self.dataset_plan["quotas_by_archetype"].items()}
        seq: list[str] = []
        while len(seq) < total and any(v > 0 for v in remaining.values()):
            for archetype in remaining:
                if remaining[archetype] <= 0:
                    continue
                seq.append(archetype)
                remaining[archetype] -= 1
                if len(seq) >= total:
                    break
        return seq

    @staticmethod
    def _fallback_narrative(case: SyntheticCase) -> str:
        d = case.diagnosis
        risk_labels = {
            "low_risk": "bajo riesgo",
            "favourable_intermediate": "riesgo intermedio favorable",
            "unfavourable_intermediate": "riesgo intermedio desfavorable",
            "high_risk_localised": "alto riesgo localizado",
            "locally_advanced": "enfermedad localmente avanzada",
            "cn1_m0": "afectacion ganglionar regional cN1 M0",
            "metastatic_hormone_sensitive_at_diagnosis": "enfermedad metastasica hormonosensible al diagnostico",
        }
        treatment_labels = {
            "active_surveillance": "vigilancia activa",
            "radical_prostatectomy": "prostatectomia radical",
            "radical_prostatectomy_plus_plnd": "prostatectomia radical con linfadenectomia pelvica",
            "external_beam_radiotherapy": "radioterapia externa",
            "external_beam_radiotherapy_plus_short_adt": "radioterapia externa con bloqueo androgenico corto",
            "external_beam_radiotherapy_plus_long_adt": "radioterapia externa con bloqueo androgenico prolongado",
            "local_treatment_plus_adt_plus_systemic_intensification": "tratamiento local asociado a bloqueo androgenico e intensificacion sistemica",
            "adt_plus_arpi": "bloqueo androgenico asociado a ARPI",
            "adt_plus_docetaxel_plus_arpi": "triplete con bloqueo androgenico, docetaxel y ARPI",
        }
        event_labels = {
            "diagnosis": "diagnostico inicial",
            "active_surveillance_follow_up": "control en vigilancia activa",
            "active_surveillance_reclassification": "reclasificacion durante vigilancia activa",
            "switch_to_curative_treatment": "cambio a tratamiento radical",
            "radical_prostatectomy": "prostatectomia radical",
            "postoperative_pathology": "revision anatomopatologica postoperatoria",
            "positive_surgical_margin": "margen quirurgico positivo",
            "external_beam_radiotherapy": "radioterapia externa",
            "androgen_deprivation_therapy": "inicio de bloqueo androgenico",
            "psa_response": "respuesta bioquimica",
            "biochemical_recurrence_after_rp": "recaida bioquimica tras prostatectomia",
            "salvage_radiotherapy": "radioterapia de rescate",
            "biochemical_recurrence_after_rt": "recaida bioquimica tras radioterapia",
            "mHSPC_response": "respuesta de enfermedad metastasica hormonosensible",
            "castration_resistant_progression": "progresion resistente a castracion",
            "mCRPC": "CPRC metastasico",
            "docetaxel": "tratamiento con docetaxel",
            "cabazitaxel": "tratamiento con cabazitaxel",
            "parp_inhibitor": "tratamiento con inhibidor PARP",
            "bone_complication": "complicacion osea",
            "palliative_care": "cuidados paliativos",
        }

        dre_label = {
            "normal": "sin nodulos sospechosos",
            "suspicious_cT2": "nodulo limitado a prostata",
            "suspicious_cT3": "sospecha de extension extracapsular",
            "suspicious_cT4": "sospecha de invasion local avanzada",
        }.get(d.dre, d.dre)

        biopsy_route = "transperineal" if d.biopsy.approach == "transperineal" else "transrectal"
        treatment = treatment_labels.get(case.treatment_plan.initial_treatment, case.treatment_plan.initial_treatment.replace("_", " "))
        risk = risk_labels.get(case.risk_classification.risk_group, case.risk_classification.risk_group)
        sites = "" if d.metastatic_sites == "none" else f" Metastasis descritas: {d.metastatic_sites.replace('_', ' ')}."

        paragraphs = [
            (
                f"Varon de {case.patient.age} anos remitido por {case.patient.presentation_context.lower()} "
                f"Antecedentes urologicos sin datos identificativos. Comorbilidad {case.patient.comorbidity_level}, "
                f"situacion funcional {case.patient.frailty}, expectativa vital {case.patient.life_expectancy_group}. "
                f"Antecedente familiar de cancer de prostata: {'si' if case.patient.family_history_pca else 'no'}. "
                f"BRCA2 {case.patient.brca2_status}."
            ),
            (
                f"PSA inicial {d.initial_psa} ng/ml"
                f"{f', repetido {d.repeat_psa} ng/ml' if d.repeat_psa is not None else ''}. "
                f"Tacto rectal {dre_label}. Volumen prostatico {d.prostate_volume_cc} cc, "
                f"densidad de PSA {d.psa_density}. RM multiparametrica realizada, PI-RADS {d.mri.pirads}, "
                f"lesion en {d.mri.lesion_location}, sospecha local {d.mri.local_stage_suspicion}. "
                f"Biopsia {biopsy_route} con estrategia {d.biopsy.strategy.replace('_', ' ')}, "
                f"{d.biopsy.positive_cores} cilindros positivos. Histologia: {d.biopsy.histology}. "
                f"Gleason {d.biopsy.gleason}, ISUP {d.biopsy.isup}"
                f"{', con patron cribiforme o intraductal' if d.biopsy.cribriform_or_intraductal else ''}. "
                f"Estadiaje clinico {d.tnm.cT} {d.tnm.cN} {d.tnm.cM}.{sites}"
            ),
            (
                f"Clasificado como {risk}. Se decide {treatment} con intencion "
                f"{case.treatment_plan.treatment_intent}. "
                f"Razonamiento registrado: {case.treatment_plan.rationale}"
            ),
        ]

        event_sentences: list[str] = []
        for event in case.progression.trajectory:
            label = event_labels.get(event.event_type, event.event_type.replace("_", " "))
            parts = [f"Mes {event.month}: {label}", f"estado {event.state}"]
            if event.psa is not None:
                parts.append(f"PSA {event.psa} ng/ml")
            if event.testosterone is not None:
                parts.append(f"testosterona {event.testosterone} ng/dl")
            if event.imaging:
                parts.append(f"imagen: {event.imaging}")
            if event.treatment:
                parts.append(f"tratamiento: {treatment_labels.get(event.treatment, event.treatment.replace('_', ' '))}")
            if event.complication:
                parts.append(f"complicacion: {event.complication}")
            if event.notes:
                parts.append(event.notes)
            event_sentences.append(", ".join(parts) + ".")

        paragraphs.append(" ".join(event_sentences))
        paragraphs.append(
            f"Situacion final: {case.progression.final_status.oncological_status}. "
            f"Seguimiento: {case.progression.final_status.follow_up_status}."
        )
        return "\n\n".join(paragraphs)
