"""Data models for experimental multimodal extraction uncertainty signals."""

from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class EvidenceStatus(str, Enum):
    """Whether an uncertainty signal is safe to use for automated routing."""

    AVAILABLE = "AVAILABLE"
    DEGENERATE = "DEGENERATE"
    UNAVAILABLE = "UNAVAILABLE"
    PARTIAL = "PARTIAL"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class MetricProvenance(BaseModel):
    """Provider and decoding details needed to interpret an uncertainty signal."""

    provider: str = Field(description="Inference provider that produced the response")
    model: str = Field(description="Provider model identifier")
    decoder: str = Field(description="Decoding or constrained-output mode")
    logprob_source: str = Field(description="Provider response field used for token likelihoods")
    logprob_quality: EvidenceStatus = Field(
        description="Availability and quality of returned logprobs"
    )
    requested_sample_count: int = Field(default=0, ge=0)
    valid_sample_count: int = Field(default=0, ge=0)


class FieldUncertainty(BaseModel):
    """Experimental uncertainty signals for one structured output field."""

    field_name: str = Field(description="Name of the Pydantic/JSON target field")
    extracted_value: Any = Field(default=None, description="Extracted value for the field")
    perplexity: float | None = Field(
        default=None,
        description="Length-normalized value-token PPL; null when token likelihoods are unavailable.",
    )
    sample_disagreement_entropy: float = Field(
        default=0.0,
        validation_alias=AliasChoices("sample_disagreement_entropy", "semantic_entropy"),
        description="Entropy of normalized sampled field strings; this is not semantic entropy.",
    )

    @property
    def semantic_entropy(self) -> float:
        """Deprecated compatibility alias for sample disagreement entropy."""
        return self.sample_disagreement_entropy


class NodeUncertainty(BaseModel):
    """Standardized node payload with evidence quality and metric provenance."""

    node_name: str = Field(description="Name or ID of the extraction agent node")
    overall_value_perplexity: float | None = Field(
        default=None,
        description="Aggregate value-token PPL; null when no trustworthy value-token likelihood exists.",
    )
    evidence_status: EvidenceStatus = Field(
        default=EvidenceStatus.UNAVAILABLE,
        description="Whether this node's uncertainty metrics may support automated routing.",
    )
    metric_provenance: MetricProvenance | None = Field(
        default=None,
        description="Provider and decoding metadata required to interpret metrics.",
    )
    calibration_id: str | None = Field(
        default=None,
        description="Identifier of the task/model/prompt calibration used for routing.",
    )
    fields: list[FieldUncertainty] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the standardized serializable node payload."""
        return self.model_dump()


class E2EEvaluatorResult(BaseModel):
    """End-to-end routing outcome for experimental extraction uncertainty signals."""

    action: str = Field(
        description=(
            "AUTO_ACCEPT, NEEDS_REVIEW, REJECT, INSUFFICIENT_EVIDENCE, or EXECUTION_FAILED"
        )
    )
    bottleneck_perplexity: float | None = None
    mean_e2e_perplexity: float | None = None
    max_field_entropy: float = 0.0
    node_results: list[NodeUncertainty] = Field(default_factory=list)


class UncertaintyConfig(BaseModel):
    """Task-specific routing configuration; thresholds require empirical calibration."""

    ppl_threshold: float = Field(default=2.2, gt=0)
    entropy_threshold: float = Field(default=0.5, ge=0)
    entropy_samples: int = Field(default=3, ge=0)
    calibration_id: str | None = None
