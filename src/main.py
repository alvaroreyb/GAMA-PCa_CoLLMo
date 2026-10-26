"""Generate synthetic longitudinal prostate cancer notes using external CK."""
import argparse
import os
from pathlib import Path

from .batch_orchestrator_agent import BatchOrchestratorAgent
from .knowledge_base import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, help='Number of positions, within configured quotas')
    parser.add_argument('--start-index', type=int)
    parser.add_argument('--dataset-plan', type=Path)
    parser.add_argument('--knowledge-dir', type=Path, help='External authorised CK; never downloaded')
    parser.add_argument('--output-dir', type=Path, help='New outputs; default: ./outputs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pending-only', action='store_true')
    parser.add_argument('--max-new-positions', type=int)
    parser.add_argument('--check', action='store_true', help='Validate CK and configuration without model calls or writes')
    args = parser.parse_args()
    if args.knowledge_dir:
        os.environ['GAMA_PCA_KNOWLEDGE_DIR'] = str(args.knowledge_dir.expanduser().resolve())
    if args.output_dir:
        os.environ['GAMA_PCA_OUTPUT_DIR'] = str(args.output_dir.expanduser().resolve())
    if args.check:
        from .config_loader import load_app_config, load_dataset_plan, load_ollama_config, load_yaml
        kb = KnowledgeBase()
        plan = load_yaml(args.dataset_plan.resolve()) if args.dataset_plan else load_dataset_plan()
        load_app_config()
        config = load_ollama_config()['ollama']
        unknown = set(plan['quotas_by_archetype']) - set(kb.archetypes)
        if unknown:
            raise ValueError('Unknown plan archetypes: ' + ', '.join(sorted(unknown)))
        if sum(plan['quotas_by_archetype'].values()) != plan['total_cases']:
            raise ValueError('Quota sum must equal total_cases')
        print(f"CK and configuration valid; generator={config['generation_model']}; no model calls")
        return
    if args.n is None and not args.resume:
        parser.error('Specify --n explicitly to start generation (model calls may incur charges)')
    orch = BatchOrchestratorAgent(
        use_llm_generation=True, start_index=args.start_index,
        dataset_plan_path=args.dataset_plan.resolve() if args.dataset_plan else None,
        knowledge_dir=args.knowledge_dir, output_dir=args.output_dir,
    )
    cases = orch.run(args.n, resume=args.resume,
                     max_new_positions=args.max_new_positions, pending_only=args.pending_only)
    print(f'Accepted cases: {len(cases)}; batch outcomes: {orch.batch_report_path}')


if __name__ == '__main__':
    main()
