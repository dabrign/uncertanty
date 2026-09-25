# Multimodal Agent Pipeline Integration (`AGENTS.md`)

This document defines the agent architecture, node contract schemas, and uncertainty propagation rules for multimodal agent pipelines using `multimodal-uncertainty`. All agent code must adhere to [style.md](style.md).

---

## Agent Node Contract

When operating in a multi-agent network (e.g., Document Router $\rightarrow$ Extraction Agent $\rightarrow$ Validation Agent), every node MUST emit a standardized `NodeUncertainty` payload alongside its structured domain response.

```json
{
  "node_name": "Document_Extraction_Agent",
  "overall_value_perplexity": 1.14,
  "fields": [
    {
      "field_name": "total_amount",
      "extracted_value": 1450.50,
      "perplexity": 1.08,
      "semantic_entropy": 0.0
    },
    {
      "field_name": "due_date",
      "extracted_value": "2026-10-15",
      "perplexity": 3.42,
      "semantic_entropy": 0.82
    }
  ]
}
```

---

## Downstream Node Behavior Rules

1. **Propagation**: If an upstream node yields `NEEDS_REVIEW`, downstream nodes MUST append an upstream warning tag to their context prompt to prevent cascading errors.
2. **Fallback Triggering**: If `bottleneck_perplexity > 3.0` or `max_field_entropy > 1.0`, the pipeline MUST suspend automated execution and forward the payload to the Human-in-the-Loop (HITL) queue.
3. **Field Isolation**: Downstream nodes should only re-prompt or perform spatial crop re-verification on fields where `perplexity > field_threshold`.
