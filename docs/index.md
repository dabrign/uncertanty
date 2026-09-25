# Multimodal Uncertainty Quantification (`multimodal-uncertainty`)

**End-to-End (E2E) Extraction Uncertainty Quantification in Multimodal Agent Pipelines.**

`multimodal-uncertainty` quantifies extraction risk across vision-language models (Gemini, Qwen2.5-VL, Llama-Vision, vLLM, and local OpenAI-compatible endpoints) by isolating **value-only log-probabilities** and computing **semantic entropy**.

---

## Quickstart

```python
from pydantic import BaseModel
from multimodal_uncertainty import GeminiUncertaintyEngine, PipelineDecisionEngine

class InvoiceSchema(BaseModel):
    total_amount: float
    vendor_name: str

engine = GeminiUncertaintyEngine()

with open("invoice.pdf", "rb") as f:
    pdf_bytes = f.read()

node_result = engine.extract_and_evaluate(
    image_or_pdf_bytes=pdf_bytes,
    mime_type="application/pdf",
    prompt="Extract invoice metadata.",
    schema_cls=InvoiceSchema
)

decision_engine = PipelineDecisionEngine(ppl_threshold=2.2, entropy_threshold=0.5)
result = decision_engine.evaluate([node_result])

print(f"Action: {result.action}")
print(f"Bottleneck Perplexity: {result.bottleneck_perplexity}")
```
