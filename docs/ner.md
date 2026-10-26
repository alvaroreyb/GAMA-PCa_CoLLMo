# Existing NER+Regex pipeline

The implementation in `src/silver/entidades_all.py` retains the existing inference, span aggregation, regex and deduplication logic. Its input directory is supplied by the isolated runner rather than a personal filesystem path.

| Family | Model identifier |
|---|---|
| BSC/PlanTL | `BSC-NLP4BIA/bsc-bio-ehr-es-livingner-species` |
| BSC/PlanTL | `BSC-NLP4BIA/bsc-bio-ehr-es-livingner-humano` |
| BSC/PlanTL | `BSC-NLP4BIA/bsc-bio-ehr-es-medprocner` |
| BSC/PlanTL | `BSC-NLP4BIA/bsc-bio-ehr-es-symptemist` |
| BSC/PlanTL | `BSC-NLP4BIA/bsc-bio-ehr-es-distemist` |
| BSC/PlanTL | `PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer` |
| MedSpaner | `medspaner/roberta-es-clinical-trials-cases-medic-attr` |
| MedSpaner | `medspaner/xlm-roberta-large-spanish-trials-cases-temp-ent` |
| MedSpaner | `medspaner/roberta-es-clinical-trials-cases-neg-spec` |

The retained confidence interval is 0.85–1.00, token stride 192, and window batches 16 (BSC/PlanTL) and eight (MedSpaner). Regex extracts PSA and Gleason without a model-confidence threshold. The tested environment uses PyTorch 2.10.0 with CUDA 12.6 and Transformers 5.5.1. CUDA availability and computation are checked before a run; CPU fallback is prohibited by the runner.

Each run copies source texts, prepares normalised annotation inputs, records hashes, executes the existing pipeline and checks the outputs. The new output directory contains `original_texts/`, `inputs/`, paired `brat/` files, model-specific outputs, a manifest and completion status. Failed runs retain evidence and are not reported as complete.

Labels may overlap; attribute spans are not a manually adjudicated relation layer. Offset checks do not establish semantic accuracy. Model identifiers are recorded; upstream model weights may change, so repository identifiers alone do not guarantee identical future predictions. Model downloads and their licenses remain the user's responsibility.
