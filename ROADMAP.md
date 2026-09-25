# Operational Roadmap: Multimodal Uncertainty Quantification

This roadmap details the planned feature trajectory for `multimodal-uncertainty`, organized by operational priority.

---

## High Priority (Core Infrastructure & Immediate Value)

- [x] **Value-Only Logprob Filtering**: Strip structural JSON syntax tokens (`{`, `}`, `:`, `,`, `"`) to isolate field value uncertainty.
- [x] **Gemini & Pydantic Engine**: Combine `response_schema` with `response_logprobs=True` using `google-genai`.
- [x] **vLLM Guided Decoding Engine**: Native implementation for open-source models using `GuidedDecodingParams`.
- [x] **Local OpenAI-Compatible Engine**: Unified support for local VLM runners (Ollama, MLX-VLM, vLLM server).
- [ ] **Field-Level UI Highlighting for HITL**: Build web components (React / Svelte / Web Components) that dynamically outline fields in red/yellow based on $PPL$ and $Entropy$ thresholds.

---

## Medium Priority (Scale & Provider Neutrality)

- [ ] **Unified Multi-Provider Abstraction via LiteLLM / OpenRouter**: Standardize logprob response structures across OpenAI, Anthropic, Gemini, and Mistral through unified wrappers.
- [ ] **FastAPI Response Middleware**: Embed uncertainty validation directly into FastAPI microservice pipelines with automated `X-Extraction-Uncertainty-Score` header generation.
- [ ] **Temperature / Platt Scaling Calibration**: Apply temperature scaling to logit outputs for fine-tuned LoRA adapters to correct probability miscalibrations.

---

## Low Priority (Advanced R&D)

- [ ] **Spatial Crop Re-Verification**: For high-entropy vision extractions, retrieve Gemini/VLM bounding box coordinates, crop the source document region, and perform targeted micro-verification.
- [ ] **Chain-of-Thought Stepwise Perplexity Tracking**: Record per-step reasoning token log-probabilities to isolate decision bottlenecks in agent CoT traces.
