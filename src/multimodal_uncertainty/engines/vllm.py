"""
vLLM Open-Source Model Uncertainty Quantification Engine.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

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


class VLLMUncertaintyEngine:
    """
    Uncertainty engine utilizing vLLM with `GuidedDecodingParams` and logprobs output.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", **vllm_kwargs: Any):
        if not HAS_VLLM:
            raise ImportError(
                "vLLM package is required for VLLMUncertaintyEngine. "
                "Install via `pip install multimodal-uncertainty[vllm]` or `pip install vllm`."
            )
        self.llm = LLM(model=model_path, trust_remote_code=True, **vllm_kwargs)
        self.model_path = model_path

    def extract_with_pydantic_logprobs(
        self, prompt: str, schema_cls: type[T], entropy_samples: int = 1
    ) -> NodeUncertainty:
        guided_params = GuidedDecodingParams(json=schema_cls.model_json_schema())
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            logprobs=1,
            guided_decoding=guided_params,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        extracted_data = schema_cls.model_validate_json(output.text)
        logprobs_data = output.logprobs or []
        target_fields = list(schema_cls.model_fields.keys())

        token_pairs = []
        for token_dict in logprobs_data:
            for token_id, logprob_obj in token_dict.items():
                token_text = logprob_obj.decoded_token
                lp = logprob_obj.logprob
                if token_text is not None and lp is not None:
                    token_pairs.append((token_text, lp))

        field_value_logprobs, all_value_logprobs = isolate_field_value_logprobs(
            token_pairs, target_fields, output.text
        )

        # Stochastic sampling if entropy requested
        samples: list[dict[str, Any]] = []
        if entropy_samples > 1:
            stochastic_params = SamplingParams(
                temperature=0.7,
                max_tokens=512,
                guided_decoding=guided_params,
                n=entropy_samples,
            )
            s_outputs = self.llm.generate([prompt], stochastic_params)
            for out in s_outputs[0].outputs:
                try:
                    parsed = schema_cls.model_validate_json(out.text)
                    samples.append(parsed.model_dump())
                except Exception:
                    pass

        field_evals = []
        for field_name in target_fields:
            probs = field_value_logprobs[field_name]
            field_ppl = calculate_length_normalized_perplexity(probs)

            vals = [s.get(field_name, "") for s in samples] if samples else []
            entropy = calculate_sample_disagreement_entropy(vals)

            field_evals.append(
                FieldUncertainty(
                    field_name=field_name,
                    extracted_value=getattr(extracted_data, field_name, None),
                    perplexity=field_ppl,
                    sample_disagreement_entropy=entropy,
                )
            )

        overall_ppl = calculate_length_normalized_perplexity(all_value_logprobs)

        if not token_pairs:
            evidence_status = EvidenceStatus.UNAVAILABLE
        elif all(logprob == 0.0 for _, logprob in token_pairs):
            evidence_status = EvidenceStatus.DEGENERATE
        elif any(not field_value_logprobs[field_name] for field_name in target_fields):
            evidence_status = EvidenceStatus.PARTIAL
        else:
            evidence_status = EvidenceStatus.AVAILABLE

        return NodeUncertainty(
            node_name="vLLM_Structured_Extraction_Node",
            overall_value_perplexity=overall_ppl,
            evidence_status=evidence_status,
            metric_provenance=MetricProvenance(
                provider="vllm",
                model=self.model_path,
                decoder="guided_decoding",
                logprob_source="RequestOutput.outputs[0].logprobs",
                logprob_quality=evidence_status,
                requested_sample_count=entropy_samples,
                valid_sample_count=len(samples),
            ),
            fields=field_evals,
        )
