# Operation

## External clinical knowledge

Extract the separately distributed GAMA-PCa CK dataset. Set `GAMA_PCA_KNOWLEDGE_DIR`, or pass `--knowledge-dir` to generation, using either the dataset root or its `clinical_knowledge/` directory. The software does not download CK automatically. The knowledge directory must contain nonempty YAML mappings in:

```text
manifest.yaml
prostate_cancer_ontology.yaml
gleason_isup_mapping.yaml
risk_groups.yaml
diagnostic_pathways.yaml
biopsy_rules.yaml
treatment_options.yaml
progression_rules.yaml
follow_up_rules.yaml
hard_constraints.yaml
archetypes.yaml
probabilities_seed.yaml
source_map.yaml
```

The runtime expects `archetypes.archetypes`, `risk_groups.risk_groups`, `gleason_isup_mapping.gleason_to_isup` and `treatment_options.allowed_treatments_by_risk_group` to be nonempty mappings. Other modules must comply with the generator schemas and executable checks. The source map records provenance; it is not a decision engine. Checking file structure does not validate clinical correctness or establish equivalence to the authors' CK version.

The CK dataset is published separately from the software and has its own license. Keep its local copy outside this repository; the original guideline PDF is not included in either package. Runtime prompts and trace files may include CK and run metadata: do not commit generated output directories.

## Batches and partial repairs

Use `--max-new-positions` to limit a run without changing its quotas. `--resume` preserves terminal outcomes and checkpoints. Existing ID ranges and outputs are protected against accidental reuse. Process locking prevents concurrent writes to the same generation directory.

Patient, diagnosis, risk, treatment and progression failures follow the explicit ownership map. Partial repairs preserve valid context and revalidate the assembled case; otherwise a fresh candidate is generated. Three identical consecutive validation failures schedule one full restart. A failed restarted candidate is recorded and the batch advances.

## Technical and structured review

Technical output failures enter `technical_review_queue/`; they are not clinical rejections. An authorised reviewer can supply a JSON manifest containing `decisions`, each with `case_id`, `case_sha256`, `decision` (`approve` or `hold`), `reason` and optionally `reviewer`. The hash is `case_digest(payload['case'])` from `src.resolve_technical_reviews`.

```bash
python -m src.resolve_technical_reviews decisions.json
python -m src.resolve_technical_reviews decisions.json --apply
python -m src.main --n 10 --start-index 1 --resume --pending-only
```

The first command is a dry run. Approval requires a matching case hash, deterministic validation and no confirmed CK contradictions. It records assisted clearance, not a completed Kimi assessment; normal narrative generation and judging still follow. `hold` retains the case for further review. The console also displays structured-review requests; it cannot override confirmed CK contradictions.

## Original and corrected narratives

Human narrative corrections are versioned separately and never silently replace originals. Select a corrected version explicitly for downstream use or NER; its judge assessment belongs to that corrected text. An initial `requires_revision` flag alone is not an instruction to rewrite a note.
