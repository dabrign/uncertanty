"""
Unit tests for engine abstractions and mock API calls.
"""

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from multimodal_uncertainty.engines.local_openai import LocalOpenAIUncertaintyEngine
from multimodal_uncertainty.models import NodeUncertainty


class SampleSchema(BaseModel):
    item_name: str
    price: float


@patch("requests.post")
def test_local_openai_uncertainty_engine_mock(mock_post):
    # Mock response matching OpenAI API logprobs format
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": '{"item_name": "Sofa", "price": 450.00}'},
                "logprobs": {
                    "content": [
                        {"token": "{", "logprob": 0.0},
                        {"token": '"item_name"', "logprob": 0.0},
                        {"token": ': "', "logprob": 0.0},
                        {"token": "Sofa", "logprob": -0.02},
                        {"token": '", "', "logprob": 0.0},
                        {"token": 'price"', "logprob": 0.0},
                        {"token": ": ", "logprob": 0.0},
                        {"token": "450.00", "logprob": -0.05},
                        {"token": "}", "logprob": 0.0},
                    ]
                },
            }
        ]
    }
    mock_post.return_value = mock_response

    engine = LocalOpenAIUncertaintyEngine(api_base="http://localhost:11434/v1", model="test-vlm")
    node_result = engine.extract_and_evaluate(
        prompt="Extract item and price",
        schema_cls=SampleSchema,
        entropy_samples=0,
    )

    assert isinstance(node_result, NodeUncertainty)
    assert node_result.overall_value_perplexity > 0.0
    assert len(node_result.fields) == 2
    assert node_result.fields[0].field_name == "item_name"
    assert node_result.fields[0].extracted_value == "Sofa"
    assert node_result.fields[1].field_name == "price"
    assert node_result.fields[1].extracted_value == 450.00

    request_payload = mock_post.call_args.kwargs["json"]
    assert request_payload["response_format"]["type"] == "json_schema"
    assert (
        request_payload["response_format"]["json_schema"]["schema"]
        == SampleSchema.model_json_schema()
    )


@patch("requests.post")
def test_local_openai_uncertainty_engine_survives_reasoning_channel_preamble(mock_post):
    """Regression for a real Ollama capture: some local backends stream a
    hidden reasoning/"thought channel" before the final JSON answer inside
    the *same* logprobs.content array, so it no longer equals message.content
    verbatim. The engine must still recover per-field perplexity from the
    tokens that actually belong to the answer."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": '{"item_name": "sofa", "price": 450.0}'},
                "logprobs": {
                    "content": [
                        {"token": "<|channel|>", "logprob": -0.01},
                        {"token": "thought", "logprob": -0.02},
                        {"token": "\n...reasoning...\n", "logprob": -0.03},
                        {"token": '{"item_name": "', "logprob": 0.0},
                        {"token": "sofa", "logprob": -0.05},
                        {"token": '", "price": ', "logprob": 0.0},
                        {"token": "450.0", "logprob": -0.08},
                        {"token": "}", "logprob": 0.0},
                    ]
                },
            }
        ]
    }
    mock_post.return_value = mock_response

    engine = LocalOpenAIUncertaintyEngine(api_base="http://localhost:11434/v1", model="test-vlm")
    node_result = engine.extract_and_evaluate(
        prompt="Extract item and price",
        schema_cls=SampleSchema,
        entropy_samples=0,
    )

    assert node_result.evidence_status == "AVAILABLE"
    assert node_result.overall_value_perplexity is not None
    assert node_result.overall_value_perplexity > 0.0
    assert node_result.fields[0].extracted_value == "sofa"
    assert node_result.fields[0].perplexity is not None
    assert node_result.fields[1].extracted_value == 450.0
    assert node_result.fields[1].perplexity is not None
