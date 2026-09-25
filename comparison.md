# Competitive Analysis: LLM & VLM Uncertainty Quantification

This document provides a comparative analysis benchmarking `multimodal-uncertainty` against existing open-source libraries and commercial platforms in the LLM uncertainty quantification, confidence scoring, and structured extraction ecosystem.

---

## Ecosystem Overview

| Feature / Dimension | `multimodal-uncertainty` | Cleanlab TLM | UQLM / LUQ | Instructor | Guardrails AI | Outlines |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Focus** | Field-level VLM Extraction UQ | LLM Trustworthiness Score | Academic UQ Metrics | Schema Enforcement | Policy Validation | Structured Decoding |
| **Value-Only Token Logprob Isolation** | ✅ **Yes** (Strips `{`, `}`, `"key":`) | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **Multimodal Vision (VLM) Support** | ✅ **Yes** (Gemini, Qwen, vLLM) | ⚠️ Text-only | ⚠️ Text-only | ⚠️ Text-only | ⚠️ Text-only | ⚠️ Text-only |
| **Semantic Entropy Sampling** | ✅ **Yes** (Kuhn et al., 2023) | ⚠️ Internal proxy | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Stepwise Bottleneck Tracking** | ✅ **Yes** (Cui et al., 2025) | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **Decision Routing Matrix** | ✅ **AUTO_ACCEPT / REVIEW / REJECT** | ⚠️ Score 0–1 | ❌ Raw metrics | ⚠️ Error raise | ⚠️ Error raise | ❌ No |
| **FastAPI Header Middleware** | ✅ **Yes** (`X-Extraction-Uncertainty-Score`) | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **Open Source License** | Apache 2.0 | Commercial / Proprietary API | MIT / Apache | MIT | Apache 2.0 | Apache 2.0 |

---

## Detailed Competitor Breakdown

### 1. Cleanlab TLM (Trustworthy Language Model)
- **Strengths**: Commercial API offering single trustworthiness scores for open-ended text generations and detecting epistemic uncertainty across black-box models.
- **Differences from `multimodal-uncertainty`**:
  - Cleanlab treats LLM responses as monolithic freeform text. It does **not** parse Pydantic schema boundaries or isolate logprobs on extracted JSON field values.
  - Proprietary API requiring paid cloud service, whereas `multimodal-uncertainty` is 100% open-source and self-hostable (vLLM, Ollama, MLX).

### 2. UQLM / LUQ (Academic UQ Toolkits)
- **Strengths**: Strong implementations of predictive entropy, mutual information, and semantic entropy for text generation.
- **Differences from `multimodal-uncertainty`**:
  - Structural Token Noise: UQLM calculates logprobs across *all* generated tokens, including structural JSON syntax (`{`, `"vendor":`). Because structural tokens have near-zero uncertainty, overall perplexity is artificially lowered, hiding high uncertainty on actual extracted values.
  - `multimodal-uncertainty` explicitly isolates field-level value tokens before calculating length-normalized perplexity.

### 3. Instructor & Guardrails AI
- **Strengths**: Excellent tools for enforcing Pydantic schema validation rules and regex constraints.
- **Differences from `multimodal-uncertainty`**:
  - Instructor and Guardrails validate **syntax and rules** (e.g., regex matching, type coercion), but cannot tell if a syntactically valid string (`"2026-10-15"`) was extracted with 99% confidence or complete hallucination.
  - `multimodal-uncertainty` complements Instructor by adding true token-level and semantic uncertainty scores to schema extractions.

### 4. Outlines / vLLM Guided Decoding
- **Strengths**: Guarantees JSON schema compliance during token generation via finite-state machine (FSM) sampling constraints.
- **Differences from `multimodal-uncertainty`**:
  - Outlines guarantees structural correctness but does **not** evaluate token logprob distributions or compute semantic entropy across stochastic generations.
  - `multimodal-uncertainty` integrates with vLLM Guided Decoding to extract token logprobs directly from the constrained sampler.

---

## Key Differentiators of `multimodal-uncertainty`

1. **Syntactic vs. Semantic Noise Isolation**: By skipping structural JSON syntax characters (`{`, `}`, `:`, `,`, `"`) and target field key tokens, `multimodal-uncertainty` calculates pure field-value log-probabilities.
2. **Multimodal Agent Pipeline Support**: First-class support for PDFs, images, and visual document models (Gemini 2.5 Flash, Qwen2.5-VL, Llama-Vision).
3. **Stepwise Chain-of-Prompts Tracking**: Tracks intermediate node perplexities to pinpoint the exact operational bottleneck in multi-agent workflows.
4. **Zero-Lock-in Infrastructure**: Works out-of-the-box with Google GenAI SDK, vLLM, Ollama, MLX-VLM, and local OpenAI-compatible endpoints.
