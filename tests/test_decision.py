"""Tests for evidence-aware routing."""

from multimodal_uncertainty.decision import PipelineDecisionEngine
from multimodal_uncertainty.models import EvidenceStatus, FieldUncertainty, NodeUncertainty


def available_node(perplexity: float, disagreement: float = 0.1) -> NodeUncertainty:
    """Build a calibrated node fixture with usable evidence."""
    return NodeUncertainty(
        node_name="Node1",
        overall_value_perplexity=perplexity,
        evidence_status=EvidenceStatus.AVAILABLE,
        fields=[
            FieldUncertainty(
                field_name="f1",
                extracted_value="val1",
                perplexity=perplexity,
                sample_disagreement_entropy=disagreement,
            )
        ],
    )


def test_decision_engine_auto_accept() -> None:
    result = PipelineDecisionEngine().evaluate([available_node(1.1)])
    assert result.action == "AUTO_ACCEPT"
    assert result.bottleneck_perplexity == 1.1


def test_decision_engine_uses_field_bottleneck() -> None:
    node = available_node(1.1)
    node.fields[0].perplexity = 2.5
    assert PipelineDecisionEngine().evaluate([node]).action == "NEEDS_REVIEW"


def test_decision_engine_rejects_invalid_output() -> None:
    node = available_node(1.1)
    node.evidence_status = EvidenceStatus.INVALID_OUTPUT
    assert PipelineDecisionEngine().evaluate([node]).action == "REJECT"


def test_decision_engine_preserves_missing_evidence() -> None:
    node = available_node(1.1)
    node.evidence_status = EvidenceStatus.DEGENERATE
    assert PipelineDecisionEngine().evaluate([node]).action == "INSUFFICIENT_EVIDENCE"
