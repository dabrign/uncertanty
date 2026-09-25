"""
Unit tests for Pydantic data models.
"""

from multimodal_uncertainty.models import (
    FieldUncertainty,
    NodeUncertainty,
    UncertaintyConfig,
)


def test_field_uncertainty_creation():
    field = FieldUncertainty(
        field_name="total_amount",
        extracted_value=120.50,
        perplexity=1.05,
        semantic_entropy=0.0,
    )
    assert field.field_name == "total_amount"
    assert field.extracted_value == 120.50
    assert field.perplexity == 1.05
    assert field.semantic_entropy == 0.0


def test_node_uncertainty_serialization():
    field = FieldUncertainty(
        field_name="vendor",
        extracted_value="Acme Corp",
        perplexity=1.12,
        semantic_entropy=0.10,
    )
    node = NodeUncertainty(
        node_name="ExtractionNode",
        overall_value_perplexity=1.12,
        fields=[field],
    )
    data = node.to_dict()
    assert data["node_name"] == "ExtractionNode"
    assert len(data["fields"]) == 1
    assert data["fields"][0]["field_name"] == "vendor"


def test_uncertainty_config_defaults():
    config = UncertaintyConfig()
    assert config.ppl_threshold == 2.2
    assert config.entropy_threshold == 0.5
    assert config.entropy_samples == 3
