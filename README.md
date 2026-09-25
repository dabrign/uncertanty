# Multimodal Uncertainty Quantification (`multimodal-uncertainty`)

[![PyPI version](https://img.shields.io/pypi/v/multimodal-uncertainty.svg)](https://pypi.org/project/multimodal-uncertainty/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI Status](https://github.com/dabrign/uncertanty/actions/workflows/ci.yml/badge.svg)](https://github.com/dabrign/uncertanty/actions)

**End-to-End (E2E) Extraction Uncertainty Quantification in Multimodal Agent Pipelines.**

`multimodal-uncertainty` quantifies extraction risk across vision-language models (Gemini, Qwen2.5-VL, Llama-Vision, vLLM, and local OpenAI-compatible endpoints) by isolating **value-only log-probabilities** and computing **semantic entropy**. Low-confidence fields can be automatically routed to Human-in-the-Loop (HITL) review or triggered for fallback verification.

---

## Key Features

- 🎯 **Value-Only Logprob Filtering**: Strips out fixed structural JSON characters (`{`, `}`, `"key":`) to eliminate false precision and compute pure extraction uncertainty.
- ⚡ **Multi-Provider Support**: Built-in engines for Google Gemini (`google-genai`), vLLM (`GuidedDecodingParams`), Ollama, MLX-VLM, and local OpenAI-compatible endpoints.
- 🧠 **Semantic Entropy Sampling**: Quantifies semantic variance across stochastic generations (Kuhn et al., 2023).
- ⚖️ **Decision Matrix Engine**: Evaluates field-level and node-level perplexities to route payloads (`AUTO_ACCEPT`, `NEEDS_REVIEW`, `REJECT`).
- 🌐 **FastAPI Middleware**: Middleware to inject uncertainty header scores (`X-Extraction-Uncertainty-Score`) into microservice responses.

---

## Scientific Foundations

1. **Semantic Uncertainty in LLMs** (*Kuhn et al., 2023*): Measures unpredictability by clustering generated answers into semantic equivalence classes rather than strict string matching.
2. **Length-Normalized Token Perplexity** (*Malinin & Gales, 2020*): Prevents joint probability penalties on long fields by normalizing per-token log probabilities.
3. **Stepwise Perplexity Reasoning** (*Cui et al., 2025*): Isolates intermediate token log-probabilities to pinpoint precise operational bottlenecks in agentic workflows.

---

## Quickstart

### Installation

```bash
pip install multimodal-uncertainty
```

For vLLM or FastAPI support:
```bash
pip install "multimodal-uncertainty[all]"
```

### Basic Gemini Usage

```python
from pydantic import BaseModel
from multimodal_uncertainty import GeminiUncertaintyEngine, PipelineDecisionEngine

class DocumentSchema(BaseModel):
    document_id: str
    total_amount: float
    vendor_name: str

# Initialize Gemini engine
engine = GeminiUncertaintyEngine()

# Read input image/pdf bytes
with open("invoice.pdf", "rb") as f:
    payload = f.read()

# Extract and evaluate field-level perplexities & semantic entropy
node_result = engine.extract_and_evaluate(
    image_or_pdf_bytes=payload,
    mime_type="application/pdf",
    prompt="Extract invoice metadata.",
    schema_cls=DocumentSchema,
)

# Route decisions
decision_engine = PipelineDecisionEngine(ppl_threshold=2.2, entropy_threshold=0.5)
result = decision_engine.evaluate([node_result])

print(f"Action: {result.action}")
print(f"Bottleneck PPL: {result.bottleneck_perplexity}")
for field in node_result.fields:
    print(f"Field: {field.field_name} | Val: {field.extracted_value} | PPL: {field.perplexity} | Entropy: {field.semantic_entropy}")
```

---

## Local VLM Testing Harness

Test uncertainty scoring locally using LM Studio, Ollama, vLLM, or MLX-VLM on a local Qwen model. The demo defaults to LM Studio and `qwen/qwen3-vl-4b`:

```bash
# In LM Studio: load qwen/qwen3-vl-4b, then start the local server (port 1234).
# This makes real requests; it does not fall back to mock output on failure.
python utils/demo_cat_dog_test.py

# Optional: use the deterministic mock demo without a server.
python utils/demo_cat_dog_test.py --mock
```

For a different local OpenAI-compatible endpoint, pass its model identifier and API base explicitly:

```bash
python utils/demo_cat_dog_test.py --api-base http://localhost:11434/v1 --model qwen2.5-vl
```

---

## Documentation & Project Portal

- 📚 **GitHub Pages Portal**: [https://dabrign.github.io/uncertanty/](https://dabrign.github.io/uncertanty/)
- 🗺️ **Roadmap**: See [ROADMAP.md](ROADMAP.md)
- 🎨 **Code Style**: See [style.md](style.md)
- 📋 **Master Execution Plan**: See [PLAN.md](PLAN.md)
- 🤝 **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- 📄 **License**: [Apache 2.0](LICENSE)
