"""
Unit tests for ChainOfPromptsEvaluator multi-step tracking.
"""

from multimodal_uncertainty.models import (
    EvidenceStatus,
    FieldUncertainty,
    NodeUncertainty,
)
from multimodal_uncertainty.pipeline import ChainOfPromptsEvaluator


def test_chain_of_prompts_evaluator_flow():
    evaluator = ChainOfPromptsEvaluator()

    node1 = NodeUncertainty(
        node_name="Node1",
        overall_value_perplexity=1.05,
        evidence_status=EvidenceStatus.AVAILABLE,
        fields=[
            FieldUncertainty(
                field_name="f1", extracted_value="a", perplexity=1.05, semantic_entropy=0.0
            )
        ],
    )

    node2 = NodeUncertainty(
        node_name="Node2",
        overall_value_perplexity=3.50,  # High PPL
        evidence_status=EvidenceStatus.AVAILABLE,
        fields=[
            FieldUncertainty(
                field_name="f2", extracted_value="b", perplexity=3.50, semantic_entropy=0.8
            )
        ],
    )

    action1 = evaluator.add_step_result(node1)
    assert action1 == "AUTO_ACCEPT"

    action2 = evaluator.add_step_result(node2)
    assert action2 == "NEEDS_REVIEW"

    # Upstream warning context should be non-empty
    warning = evaluator.get_upstream_warning_context()
    assert "UPSTREAM UNCERTAINTY WARNING" in warning

    # Identify bottleneck
    node_name, max_ppl = evaluator.identify_bottleneck_step()
    assert node_name == "Node2"
    assert max_ppl == 3.50
