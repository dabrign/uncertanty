"""
Google Gemini Vision-Language Model Uncertainty Quantification Engine.
"""

import math
import os
from typing import Any, Dict, List, Type, TypeVar
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types

    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

from multimodal_uncertainty.filtering import (
    calculate_length_normalized_perplexity,
    calculate_sample_disagreement_entropy,
    isolate_field_value_logprobs,
)
from multimodal_uncertainty.models import (
    EvidenceStatus,
    FieldUncertainty,
    MetricProvenance,
    NodeUncertainty,
)

T = TypeVar("T", bound=BaseModel)


class GeminiUncertaintyEngine:
    """
    Uncertainty engine utilizing Google GenAI SDK (`google-genai`) with `response_schema`
    and `response_logprobs=True`.
    """

    def __init__(self, api_key: str | None = None):
        if not HAS_GEMINI_SDK:
            raise ImportError(
                "Google GenAI SDK is required. Install via `pip install google-genai`."
            )
        api_key_clean = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key_clean)

    def extract_and_evaluate(
        self,
        image_or_pdf_bytes: bytes,
        mime_type: str,
        prompt: str,
        schema_cls: Type[T],
        model_name: str = "gemini-2.5-flash",
        entropy_samples: int = 3,
    ) -> NodeUncertainty:
        target_fields = list(schema_cls.model_fields.keys())

        # 1. Single-pass extraction with logprobs
        response = self.client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_or_pdf_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=0.0,
                response_logprobs=True,
                logprobs=1,
            ),
        )

        extracted_pydantic = response.parsed
        candidate = response.candidates[0]
        logprob_result = candidate.logprobs_result

        token_pairs = []
        if logprob_result and logprob_result.chosen_candidates:
            for item in logprob_result.chosen_candidates:
                token_pairs.append((item.token, item.log_probability))

        field_value_logprobs, all_value_logprobs = isolate_field_value_logprobs(
            token_pairs, target_fields, response.text
        )

        # 2. Multi-sample generation for Semantic Entropy
        samples: List[Dict[str, Any]] = []
        for _ in range(entropy_samples):
            res = self.client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_or_pdf_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_cls,
                    temperature=0.7,
                ),
            )
            if hasattr(res, "parsed") and res.parsed:
                samples.append(res.parsed.model_dump())

        # 3. Compute metrics per field
        field_evals = []
        for field_name in target_fields:
            probs = field_value_logprobs[field_name]
            field_ppl = calculate_length_normalized_perplexity(probs)

            vals = [s.get(field_name, "") for s in samples]
            entropy = calculate_sample_disagreement_entropy(vals)

            val = getattr(extracted_pydantic, field_name, None) if extracted_pydantic else None
            field_evals.append(
                FieldUncertainty(
                    field_name=field_name,
                    extracted_value=val,
                    perplexity=field_ppl,
                    sample_disagreement_entropy=entropy,
                )
            )

        overall_ppl = calculate_length_normalized_perplexity(all_value_logprobs)

        if not extracted_pydantic:
            evidence_status = EvidenceStatus.INVALID_OUTPUT
        elif not token_pairs:
            evidence_status = EvidenceStatus.UNAVAILABLE
        elif all(logprob == 0.0 for _, logprob in token_pairs):
            evidence_status = EvidenceStatus.DEGENERATE
        elif any(not field_value_logprobs[field_name] for field_name in target_fields):
            evidence_status = EvidenceStatus.PARTIAL
        else:
            evidence_status = EvidenceStatus.AVAILABLE

        return NodeUncertainty(
            node_name="Gemini_Structured_Extraction_Node",
            overall_value_perplexity=overall_ppl,
            evidence_status=evidence_status,
            metric_provenance=MetricProvenance(
                provider="gemini",
                model=model_name,
                decoder="response_schema",
                logprob_source="candidates[0].logprobs_result.chosen_candidates",
                logprob_quality=evidence_status,
                requested_sample_count=entropy_samples,
                valid_sample_count=len(samples),
            ),
            fields=field_evals,
        )
