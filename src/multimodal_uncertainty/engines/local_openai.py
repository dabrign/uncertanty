"""
Universal Local OpenAI-compatible VLM Engine (Ollama, MLX-VLM, vLLM Server, LM Studio).
"""

import json
from typing import Any, TypeVar

import requests
from pydantic import BaseModel

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


class LocalOpenAIUncertaintyEngine:
    """
    Uncertainty engine interacting with local OpenAI-compatible endpoints
    (e.g., Ollama `http://localhost:11434/v1`, vLLM `http://localhost:8000/v1`, MLX server).
    """

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "qwen2.5-vl",
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def extract_and_evaluate(
        self,
        prompt: str,
        schema_cls: type[T],
        image_base64: str | None = None,
        entropy_samples: int = 3,
    ) -> NodeUncertainty:
        target_fields = list(schema_cls.model_fields.keys())

        # Construct message payload
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        # System prompt instructing JSON output matching schema
        system_prompt = (
            f"You are a structured extraction engine. You MUST respond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema_cls.model_json_schema(), indent=2)}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__.lower(),
                    "strict": True,
                    "schema": schema_cls.model_json_schema(),
                },
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 1. Primary extraction call
        resp = requests.post(
            f"{self.api_base}/chat/completions", json=payload, headers=headers, timeout=60
        )
        resp.raise_for_status()
        res_data = resp.json()

        choice = res_data["choices"][0]
        extracted_text = choice["message"]["content"]

        try:
            parsed_data = schema_cls.model_validate_json(extracted_text)
        except ValueError:
            parsed_data = None

        # Parse logprobs if provided by local endpoint
        logprob_info = choice.get("logprobs")
        token_pairs = []
        if logprob_info and "content" in logprob_info:
            for item in logprob_info["content"]:
                t_str = item.get("token", "")
                lp = item.get("logprob")
                if lp is not None:
                    token_pairs.append((t_str, lp))

        field_value_logprobs, all_value_logprobs = isolate_field_value_logprobs(
            token_pairs, target_fields, extracted_text
        )

        # 2. Multi-sample generation for Semantic Entropy
        samples: list[dict[str, Any]] = []
        sample_failures = 0
        if entropy_samples > 0:
            sample_payload = dict(payload)
            sample_payload["temperature"] = 0.7
            for _ in range(entropy_samples):
                try:
                    s_resp = requests.post(
                        f"{self.api_base}/chat/completions",
                        json=sample_payload,
                        headers=headers,
                        timeout=60,
                    )
                    s_resp.raise_for_status()
                    s_text = s_resp.json()["choices"][0]["message"]["content"]
                    s_parsed = schema_cls.model_validate_json(s_text)
                    samples.append(s_parsed.model_dump())
                except (KeyError, ValueError, requests.RequestException):
                    sample_failures += 1

        if parsed_data is None:
            evidence_status = EvidenceStatus.INVALID_OUTPUT
        elif not token_pairs:
            evidence_status = EvidenceStatus.UNAVAILABLE
        elif all(logprob == 0.0 for _, logprob in token_pairs):
            evidence_status = EvidenceStatus.DEGENERATE
        elif (
            any(not field_value_logprobs[field_name] for field_name in target_fields)
            or sample_failures
        ):
            evidence_status = EvidenceStatus.PARTIAL
        else:
            evidence_status = EvidenceStatus.AVAILABLE

        # 3. Calculate metrics per field
        field_evals = []
        for field_name in target_fields:
            probs = field_value_logprobs[field_name]
            field_ppl = calculate_length_normalized_perplexity(probs)

            vals = [s.get(field_name, "") for s in samples] if samples else []
            entropy = calculate_sample_disagreement_entropy(vals)

            val = getattr(parsed_data, field_name, None) if parsed_data else None
            field_evals.append(
                FieldUncertainty(
                    field_name=field_name,
                    extracted_value=val,
                    perplexity=field_ppl,
                    sample_disagreement_entropy=entropy,
                )
            )

        overall_ppl = calculate_length_normalized_perplexity(all_value_logprobs)

        return NodeUncertainty(
            node_name=f"Local_{self.model}_Extraction_Node",
            overall_value_perplexity=overall_ppl,
            evidence_status=evidence_status,
            metric_provenance=MetricProvenance(
                provider="local_openai_compatible",
                model=self.model,
                decoder="json_schema",
                logprob_source="choices[0].logprobs.content",
                logprob_quality=evidence_status,
                requested_sample_count=entropy_samples,
                valid_sample_count=len(samples),
            ),
            fields=field_evals,
        )
