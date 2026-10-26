from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from .config_loader import PROJECT_ROOT, load_yaml


class KnowledgeBase:
    REQUIRED_FILES = (
        "manifest", "prostate_cancer_ontology", "gleason_isup_mapping",
        "risk_groups", "diagnostic_pathways", "biopsy_rules", "treatment_options",
        "progression_rules", "follow_up_rules", "hard_constraints", "archetypes",
        "probabilities_seed", "source_map",
    )

    def __init__(self, root: Path = PROJECT_ROOT, knowledge_dir: Path | None = None):
        self.root = root
        configured = knowledge_dir or os.environ.get("GAMA_PCA_KNOWLEDGE_DIR")
        if not configured:
            raise FileNotFoundError(
                "Clinical knowledge is distributed as a separate dataset, not bundled with the software. "
                "Extract the CK dataset and supply --knowledge-dir or GAMA_PCA_KNOWLEDGE_DIR."
            )
        self.knowledge_dir = Path(configured).expanduser().resolve()
        nested = self.knowledge_dir / "clinical_knowledge"
        if not (self.knowledge_dir / "manifest.yaml").is_file() and nested.is_dir():
            self.knowledge_dir = nested
        self.data: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        missing = [name + ".yaml" for name in self.REQUIRED_FILES
                   if not (self.knowledge_dir / (name + ".yaml")).is_file()]
        if missing:
            raise FileNotFoundError("Incomplete external CK: " + ", ".join(missing))
        for name in self.REQUIRED_FILES:
            self.data[name] = load_yaml(self.knowledge_dir / (name + ".yaml"))
            if not self.data[name]:
                raise ValueError(f"Empty external CK mapping: {name}.yaml")
        for name, key in (
            ("archetypes", "archetypes"), ("risk_groups", "risk_groups"),
            ("gleason_isup_mapping", "gleason_to_isup"),
            ("treatment_options", "allowed_treatments_by_risk_group"),
        ):
            if not isinstance(self.data[name].get(key), dict) or not self.data[name][key]:
                raise ValueError(f"External CK requires a nonempty mapping: {name}.yaml:{key}")

    def get(self, name: str) -> dict[str, Any]:
        if name not in self.data:
            raise KeyError(f"Knowledge file not loaded: {name}")
        return self.data[name]

    @property
    def archetypes(self) -> dict[str, Any]:
        return self.get("archetypes").get("archetypes", {})

    def archetype(self, name: str) -> dict[str, Any]:
        return self.archetypes[name]

    def gleason_to_isup(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.get("gleason_isup_mapping").get("gleason_to_isup", {}).items()}

    def allowed_treatments(self, risk_group: str) -> set[str]:
        groups = self.get("treatment_options").get("allowed_treatments_by_risk_group", {})
        values = groups.get(risk_group, {})
        allowed: set[str] = set()
        for key in ("preferred", "alternatives", "selected_cases"):
            allowed.update(values.get(key, []) or [])
        return allowed
