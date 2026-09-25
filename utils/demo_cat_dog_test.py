"""
Demo Benchmark Script: Cat vs. Dog Uncertainty Quantification.

Tests two different prompts on the synthetic image of a cat on a sofa against an
OpenAI-compatible local vision-language endpoint (LM Studio by default):
1. "Where is the cat?" (Valid target -> Expected: Low PPL & low Entropy -> AUTO_ACCEPT)
2. "Where is the dog?" (Out-of-domain / Hallucinated target -> Expected: High PPL & high Entropy -> REJECT/NEEDS_REVIEW)
"""

import argparse
import base64

from generate_test_image import generate_cat_on_sofa_image
from pydantic import BaseModel, Field

from multimodal_uncertainty.decision import PipelineDecisionEngine
from multimodal_uncertainty.engines.local_openai import LocalOpenAIUncertaintyEngine
from multimodal_uncertainty.models import FieldUncertainty, NodeUncertainty


class ObjectLocationSchema(BaseModel):
    is_present: bool = Field(description="Whether the queried object is present in the image")
    location_description: str = Field(description="Exact location of the object if present")
    confidence_reasoning: str = Field(description="Reasoning for detection")


def run_mock_demo() -> None:
    """Runs a simulated benchmark demonstration illustrating the metrics output."""
    print("==========================================================================")
    print("  RUNNING MOCK CAT vs. DOG UNCERTAINTY DEMONSTRATION")
    print("==========================================================================")

    # 1. Valid Query: Where is the cat?
    node_cat = NodeUncertainty(
        node_name="Local_VLM_Cat_Query_Node",
        overall_value_perplexity=1.05,
        fields=[
            FieldUncertainty(
                field_name="is_present", extracted_value=True, perplexity=1.01, semantic_entropy=0.0
            ),
            FieldUncertainty(
                field_name="location_description",
                extracted_value="resting on the blue sofa cushion",
                perplexity=1.08,
                semantic_entropy=0.15,
            ),
            FieldUncertainty(
                field_name="confidence_reasoning",
                extracted_value="orange cat clearly visible in center",
                perplexity=1.06,
                semantic_entropy=0.08,
            ),
        ],
    )

    # 2. Hallucination Query: Where is the dog?
    node_dog = NodeUncertainty(
        node_name="Local_VLM_Dog_Query_Node",
        overall_value_perplexity=4.82,
        fields=[
            FieldUncertainty(
                field_name="is_present",
                extracted_value=False,
                perplexity=3.12,
                semantic_entropy=0.92,
            ),
            FieldUncertainty(
                field_name="location_description",
                extracted_value="not visible or under the sofa",
                perplexity=5.41,
                semantic_entropy=1.45,
            ),
            FieldUncertainty(
                field_name="confidence_reasoning",
                extracted_value="unable to detect canine features in image",
                perplexity=4.22,
                semantic_entropy=1.10,
            ),
        ],
    )

    decision_engine = PipelineDecisionEngine(ppl_threshold=2.2, entropy_threshold=0.5)

    res_cat = decision_engine.evaluate([node_cat])
    res_dog = decision_engine.evaluate([node_dog])

    print("\n--- QUERY 1: 'Where is the cat?' (Valid Query) ---")
    print(f"Action:                 {res_cat.action}")
    print(f"Bottleneck Perplexity: {res_cat.bottleneck_perplexity}")
    print(f"Max Field Entropy:     {res_cat.max_field_entropy}")
    for f in node_cat.fields:
        print(
            f"  Field: {f.field_name:<22} | Val: {f.extracted_value!s:<35} | PPL: {f.perplexity:<5} | SE: {f.semantic_entropy}"
        )

    print("\n--- QUERY 2: 'Where is the dog?' (Hallucinated/Absent Query) ---")
    print(f"Action:                 {res_dog.action}")
    print(f"Bottleneck Perplexity: {res_dog.bottleneck_perplexity}")
    print(f"Max Field Entropy:     {res_dog.max_field_entropy}")
    for f in node_dog.fields:
        print(
            f"  Field: {f.field_name:<22} | Val: {f.extracted_value!s:<35} | PPL: {f.perplexity:<5} | SE: {f.semantic_entropy}"
        )

    print("\n==========================================================================")
    print("  CONCLUSION: The decision engine successfully auto-accepted the valid query")
    print("  and flagged/rejected the out-of-domain prompt injection query!")
    print("==========================================================================")


def print_result(
    query: str, node: NodeUncertainty, action: str, bottleneck: float, entropy: float
) -> None:
    """Print a live node payload and its pipeline-routing decision.

    Args:
        query: Natural-language query sent to the VLM.
        node: Standardized uncertainty payload returned by the library.
        action: Decision-engine action for the payload.
        bottleneck: Highest perplexity used in the routing decision.
        entropy: Highest field semantic entropy used in the routing decision.
    """
    print(f"\n--- QUERY: {query!r} ---")
    print(node.model_dump_json(indent=2))
    print(f"Decision: {action} | Bottleneck PPL: {bottleneck} | Max entropy: {entropy}")


def run_live_demo(
    api_base: str,
    api_key: str,
    model: str,
    image_path: str,
    entropy_samples: int,
) -> None:
    """Run the cat-versus-dog benchmark against a local OpenAI-compatible VLM.

    Args:
        api_base: Base URL of the local server's OpenAI-compatible API.
        api_key: API key accepted by the local server; LM Studio accepts any non-empty value.
        model: Loaded vision model identifier reported by the local server.
        image_path: Destination for the generated benchmark image.
        entropy_samples: Number of stochastic generations for semantic entropy.
    """
    print(f"Connecting to live VLM endpoint at {api_base} with model {model}...")
    img_bytes = generate_cat_on_sofa_image(image_path)
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    engine = LocalOpenAIUncertaintyEngine(api_base=api_base, api_key=api_key, model=model)
    decision_engine = PipelineDecisionEngine(ppl_threshold=2.2, entropy_threshold=0.5)

    print("\n[Executing Query 1]: 'Where is the cat?'")
    node_cat = engine.extract_and_evaluate(
        prompt="Identify if a cat is present and describe its location.",
        schema_cls=ObjectLocationSchema,
        image_base64=img_b64,
        entropy_samples=entropy_samples,
    )
    res_cat = decision_engine.evaluate([node_cat])
    print_result(
        "Where is the cat?",
        node_cat,
        res_cat.action,
        res_cat.bottleneck_perplexity,
        res_cat.max_field_entropy,
    )

    print("\n[Executing Query 2]: 'Where is the dog?'")
    node_dog = engine.extract_and_evaluate(
        prompt="Identify if a dog is present and describe its location.",
        schema_cls=ObjectLocationSchema,
        image_base64=img_b64,
        entropy_samples=entropy_samples,
    )
    res_dog = decision_engine.evaluate([node_dog])
    print_result(
        "Where is the dog?",
        node_dog,
        res_dog.action,
        res_dog.bottleneck_perplexity,
        res_dog.max_field_entropy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat vs Dog local VLM uncertainty benchmark")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run deterministic demonstration data instead of calling a local VLM.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:1234/v1",
        help="OpenAI-compatible base URL (LM Studio default: http://localhost:1234/v1).",
    )
    parser.add_argument("--api-key", default="lm-studio", help="API key for the local server.")
    parser.add_argument(
        "--model",
        default="qwen/qwen3-vl-4b",
        help="Loaded model identifier (LM Studio: qwen/qwen3-vl-4b).",
    )
    parser.add_argument(
        "--image-path",
        default="utils/cat_on_sofa.jpg",
        help="Path where the generated benchmark image will be written.",
    )
    parser.add_argument(
        "--entropy-samples",
        type=int,
        default=2,
        help="Number of temperature-0.7 samples used for semantic entropy.",
    )
    args = parser.parse_args()

    if args.entropy_samples < 0:
        parser.error("--entropy-samples must be zero or greater")

    if args.mock:
        run_mock_demo()
    else:
        try:
            run_live_demo(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                image_path=args.image_path,
                entropy_samples=args.entropy_samples,
            )
        except Exception as error:
            raise SystemExit(
                "Live benchmark failed; no mock fallback was used. Ensure LM Studio's local server is "
                f"running, the model is loaded, and the model ID is correct. Details: {error}"
            ) from error
