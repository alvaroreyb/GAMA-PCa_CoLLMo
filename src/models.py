from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Comorbidity = Literal["low", "moderate", "high"]
Frailty = Literal["fit", "vulnerable", "frail"]
LifeExpectancy = Literal["<10y", "10-15y", ">15y"]
DRE = Literal["normal", "suspicious_cT2", "suspicious_cT3", "suspicious_cT4"]
MetastaticSites = Literal["none", "pelvic_nodes", "distant_nodes", "bone", "visceral", "bone_and_nodes", "multiple"]
TreatmentIntent = Literal["curative", "disease_control", "palliative", "surveillance"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatientProfile(StrictModel):
    age: int = Field(ge=40, le=100)
    sex: Literal["male"] = "male"
    comorbidity_level: Comorbidity
    frailty: Frailty
    life_expectancy_group: LifeExpectancy
    family_history_pca: bool
    brca2_status: Literal["unknown", "negative", "positive"]
    presentation_context: str


class MRI(StrictModel):
    performed: bool
    pirads: int | None = Field(default=None, ge=1, le=5)
    lesion_location: str | None = None
    local_stage_suspicion: str | None = None


class Biopsy(StrictModel):
    performed: bool
    approach: Literal["transperineal", "transrectal", "none"]
    strategy: Literal["systematic", "targeted", "targeted_plus_perilesional", "limited_direct", "none"]
    histology: str | None = None
    gleason: str | None = None
    isup: int | None = Field(default=None, ge=1, le=5)
    positive_cores: int | None = Field(default=None, ge=0, le=30)
    tumour_extent: str | None = None
    cribriform_or_intraductal: bool = False
    lymphovascular_invasion_if_present: str | None = None
    extraprostatic_extension_if_present: str | None = None


class TNM(StrictModel):
    cT: str
    cN: str
    cM: str


class Diagnosis(StrictModel):
    initial_psa: float = Field(gt=0)
    repeat_psa: float | None = Field(default=None, gt=0)
    dre: DRE
    prostate_volume_cc: float = Field(gt=0)
    psa_density: float = Field(ge=0)
    mri: MRI
    biopsy: Biopsy
    tnm: TNM
    metastatic_sites: MetastaticSites


class RiskClassification(StrictModel):
    risk_group: str
    matched_rule: str
    explanation: str


class TreatmentPlan(StrictModel):
    initial_treatment: str
    treatment_intent: TreatmentIntent
    treatment_components: list[str] = Field(default_factory=list)
    rationale: str


class ClinicalEvent(StrictModel):
    month: int = Field(ge=0)
    event_type: str
    state: str
    psa: float | None = Field(default=None, ge=0)
    testosterone: float | None = Field(default=None, ge=0)
    imaging: str | None = None
    treatment: str | None = None
    complication: str | None = None
    notes: str | None = None
    systemic_staging_modality: str | None = None
    systemic_staging_result: str | None = None


class FinalStatus(StrictModel):
    oncological_status: str
    follow_up_status: str


class ProgressionPlan(StrictModel):
    trajectory: list[ClinicalEvent]
    final_status: FinalStatus


class SyntheticCase(StrictModel):
    case_id: str
    archetype: str
    patient: PatientProfile
    diagnosis: Diagnosis
    risk_classification: RiskClassification
    treatment_plan: TreatmentPlan
    progression: ProgressionPlan
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(StrictModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KimiReviewResult(StrictModel):
    passed: bool
    clinical_plausibility_score: int = Field(default=10, ge=0, le=10)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


JUDGE_DOMAIN_KEYS = (
    "diagnostic_coherence",
    "risk_coherence",
    "staging_coherence",
    "therapeutic_coherence",
    "temporal_coherence",
    "followup_coherence",
    "progression_coherence",
    "absence_of_contradictions",
    "clinical_documentary_realism",
    "clinical_safety",
)


class ScoreWithComment(StrictModel):
    score: int = Field(ge=1, le=5)
    comment: str


class JudgeResult(StrictModel):
    scores: dict[str, ScoreWithComment]
    critical_error: bool
    clinically_acceptable: bool
    requires_revision: bool
    general_observations: str

    @model_validator(mode="after")
    def validate_domains_and_flags(self):
        actual = set(self.scores)
        expected = set(JUDGE_DOMAIN_KEYS)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Invalid judge domains; missing={missing}, extra={extra}")
        active_flags = sum(
            (self.critical_error, self.clinically_acceptable, self.requires_revision)
        )
        if active_flags > 1:
            raise ValueError("Only one narrative-review binary classification may be true")
        return self

    @property
    def mean_score(self) -> float:
        return sum(item.score for item in self.scores.values()) / len(self.scores)


class CaseMetadata(StrictModel):
    case_id: str
    archetype: str
    risk_group: str
    initial_treatment: str
    has_recurrence: bool
    has_metastasis: bool
    has_crpc: bool
    has_palliative_care: bool
    n_events: int
    deterministic_validation_passed: bool
    kimi_review_passed: bool | None = None
    judge_mean_score: float | None = None
    judge_critical_error: bool | None = None
    judge_clinically_acceptable: bool | None = None
    judge_requires_revision: bool | None = None


class PatientBlock(PatientProfile):
    pass
