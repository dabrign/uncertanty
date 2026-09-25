"""Benchmark receipt extraction uncertainty with mock, local, or Gemini VLMs."""

import argparse
import base64
from io import BytesIO
from typing import Literal

from pydantic import BaseModel, Field

from multimodal_uncertainty.decision import PipelineDecisionEngine
from multimodal_uncertainty.engines.gemini import GeminiUncertaintyEngine
from multimodal_uncertainty.engines.local_openai import LocalOpenAIUncertaintyEngine
from multimodal_uncertainty.models import NodeUncertainty

try:
    from datasets import load_dataset

    HAS_HUGGINGFACE_DATASETS = True
except ImportError:
    HAS_HUGGINGFACE_DATASETS = False

EngineName = Literal["mock", "local", "gemini"]


class SROIEReceiptSchema(BaseModel):
    """Structured receipt fields used for the SROIE extraction benchmark."""

    company: str = Field(description="Company or store name")
    date: str = Field(description="Transaction date")
    address: str = Field(description="Store address")
    total: float = Field(description="Total transaction amount")


def image_to_jpeg_bytes(image: object) -> bytes:
    """Serialize a Pillow-compatible dataset image as RGB JPEG bytes.

    Args:
        image: Image value returned by the Hugging Face dataset.

    Returns:
        JPEG-encoded image bytes.

    Raises:
        TypeError: If the dataset sample does not expose Pillow's ``save`` method.
    """
    if not hasattr(image, "save"):
        raise TypeError("Dataset sample does not contain a Pillow-compatible `image` field.")

    normalized_image = image.convert("RGB") if hasattr(image, "convert") else image
    buffer = BytesIO()
    normalized_image.save(buffer, format="JPEG")
    return buffer.getvalue()


def create_engine(engine_name: EngineName, api_base: str, api_key: str, model: str) -> object:
    """Create the requested live uncertainty engine.

    Args:
        engine_name: Provider to instantiate; mock is handled separately.
        api_base: Base URL for a local OpenAI-compatible server.
        api_key: Gemini key or local-server bearer value.
        model: Provider model identifier.

    Returns:
        Initialized local or Gemini uncertainty engine.

    Raises:
        ValueError: If asked to create an engine for mock mode.
    """
    if engine_name == "local":
        return LocalOpenAIUncertaintyEngine(api_base=api_base, api_key=api_key, model=model)
    if engine_name == "gemini":
        return GeminiUncertaintyEngine(api_key=api_key or None)
    raise ValueError("Mock mode does not use a live uncertainty engine.")


def evaluate_sample(
    engine: object,
    engine_name: EngineName,
    image_bytes: bytes,
    model: str,
    entropy_samples: int,
) -> NodeUncertainty:
    """Extract receipt fields and calculate uncertainty for one dataset image.

    Args:
        engine: Initialized provider engine.
        engine_name: Provider associated with the engine.
        image_bytes: JPEG bytes for the current receipt.
        model: Gemini model identifier; unused by local engines after initialization.
        entropy_samples: Number of stochastic generations for semantic entropy.

    Returns:
        Standardized node uncertainty payload.
    """
    prompt = "Extract company, date, address, and total amount from this receipt."
    if engine_name == "local":
        if not isinstance(engine, LocalOpenAIUncertaintyEngine):
            raise TypeError("Local benchmark requires LocalOpenAIUncertaintyEngine.")
        return engine.extract_and_evaluate(
            prompt=prompt,
            schema_cls=SROIEReceiptSchema,
            image_base64=base64.b64encode(image_bytes).decode("utf-8"),
            entropy_samples=entropy_samples,
        )

    if engine_name == "gemini":
        if not isinstance(engine, GeminiUncertaintyEngine):
            raise TypeError("Gemini benchmark requires GeminiUncertaintyEngine.")
        return engine.extract_and_evaluate(
            image_or_pdf_bytes=image_bytes,
            mime_type="image/jpeg",
            prompt=prompt,
            schema_cls=SROIEReceiptSchema,
            model_name=model,
            entropy_samples=entropy_samples,
        )

    raise ValueError("Mock mode does not evaluate live samples.")


def run_dataset_benchmark(
    dataset_name: str = "jsdnrs/ICDAR2019-SROIE",
    split: str = "test",
    num_samples: int = 5,
    engine_name: EngineName = "mock",
    api_base: str = "http://localhost:1234/v1",
    api_key: str = "",
    model: str = "qwen/qwen3-vl-4b",
    entropy_samples: int = 2,
) -> None:
    """Run a receipt benchmark using simulated data or a selected live provider.

    Live modes never fall back to mock output: a provider, dataset, or extraction
    error terminates the benchmark so its results cannot be mistaken for live data.

    Args:
        dataset_name: Hugging Face dataset repository identifier.
        split: Dataset split to load.
        num_samples: Maximum number of examples to evaluate.
        engine_name: ``mock``, ``local``, or ``gemini``.
        api_base: Local OpenAI-compatible endpoint URL.
        api_key: Optional Gemini key or local API bearer value.
        model: Model identifier for the selected provider.
        entropy_samples: Stochastic samples per extraction for semantic entropy.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    if entropy_samples < 0:
        raise ValueError("entropy_samples must be zero or greater")

    print("=" * 74)
    print(
        f"  DATASET BENCHMARK [{engine_name.upper()}]: {dataset_name} ({split}, {num_samples} samples)"
    )
    print("=" * 74)

    if engine_name == "mock":
        run_simulated_dataset_benchmark(num_samples)
        return

    if not HAS_HUGGINGFACE_DATASETS:
        raise ImportError(
            "Live dataset benchmarks require `datasets` and Pillow. Install with "
            '`pip install "multimodal-uncertainty[dev]" datasets`. '
        )

    print(f"Loading dataset `{dataset_name}`...")
    dataset = load_dataset(dataset_name, split=split)
    engine = create_engine(engine_name, api_base, api_key, model)
    decision_engine = PipelineDecisionEngine(ppl_threshold=2.2, entropy_threshold=0.5)
    actions = {"AUTO_ACCEPT": 0, "NEEDS_REVIEW": 0, "REJECT": 0}
    processed = 0

    for sample_number, item in enumerate(dataset, start=1):
        if processed >= num_samples:
            break

        image_bytes = image_to_jpeg_bytes(item.get("image"))
        node_result = evaluate_sample(engine, engine_name, image_bytes, model, entropy_samples)
        decision = decision_engine.evaluate([node_result])
        actions[decision.action] += 1
        processed += 1

        print(f"\n--- Sample #{sample_number} ---")
        print(node_result.model_dump_json(indent=2))
        print(
            f"Decision: {decision.action} | Bottleneck PPL: {decision.bottleneck_perplexity} "
            f"| Max SE: {decision.max_field_entropy}"
        )

    if processed == 0:
        raise ValueError(f"Dataset `{dataset_name}` split `{split}` did not contain any samples.")

    print("\n" + "=" * 74)
    print(f"BENCHMARK RESULTS SUMMARY ({processed} live samples):")
    for action, count in actions.items():
        print(f"  {action}: {count:>3} ({count / processed * 100:.1f}%)")
    print("=" * 74)


def run_simulated_dataset_benchmark(num_samples: int = 5) -> None:
    """Print deterministic sample results without downloading data or calling a model."""
    print("\n[Simulated SROIE Receipt Evaluation]")
    samples = [
        ("Clean Receipt #101", 1.04, 0.0, "AUTO_ACCEPT"),
        ("Clean Receipt #102", 1.12, 0.05, "AUTO_ACCEPT"),
        ("Blurry Address Receipt #103", 2.65, 0.62, "NEEDS_REVIEW"),
        ("Creased Receipt #104", 1.18, 0.10, "AUTO_ACCEPT"),
        ("Faded Ink Receipt #105", 4.90, 1.35, "REJECT"),
    ]
    for name, perplexity, entropy, action in samples[:num_samples]:
        print(f"  Sample: {name:<28} | PPL: {perplexity:<5} | SE: {entropy:<5} -> {action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt extraction uncertainty benchmark")
    parser.add_argument(
        "--dataset", default="jsdnrs/ICDAR2019-SROIE", help="Hugging Face dataset name"
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument(
        "--engine",
        choices=("mock", "local", "gemini"),
        default="mock",
        help="Provider to run; mock is deterministic and makes no network requests.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:1234/v1",
        help="Local OpenAI-compatible endpoint (LM Studio default).",
    )
    parser.add_argument("--api-key", default="", help="Local bearer value or Gemini API key.")
    parser.add_argument(
        "--model",
        default="qwen/qwen3-vl-4b",
        help="Local model ID or Gemini model name (for example, gemini-2.5-flash).",
    )
    parser.add_argument(
        "--entropy-samples", type=int, default=2, help="Stochastic generations per image."
    )
    arguments = parser.parse_args()

    run_dataset_benchmark(
        dataset_name=arguments.dataset,
        split=arguments.split,
        num_samples=arguments.samples,
        engine_name=arguments.engine,
        api_base=arguments.api_base,
        api_key=arguments.api_key,
        model=arguments.model,
        entropy_samples=arguments.entropy_samples,
    )
