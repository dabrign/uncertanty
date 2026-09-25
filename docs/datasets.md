# Testing on Public Multimodal Datasets

`multimodal-uncertainty` includes built-in benchmarking utilities for evaluating extraction uncertainty on standard public vision datasets from Hugging Face Hub:

1. **SROIE (Scanned Receipts OCR and Information Extraction)**: `jsdnrs/ICDAR2019-SROIE`
2. **CORD (Consolidated Receipt Dataset)**: `naver-clova-ix/cord-v2`
3. **DocVQA (Document Visual Question Answering)**: `HuggingFaceM4/DocumentVQA`

---

## Running Dataset Benchmarks

Install `datasets` and `pillow`:
```bash
pip install datasets pillow
```

Run the deterministic mock benchmark (the default; no dataset download or model call):
```bash
python utils/dataset_benchmark.py --samples 10
```

Run SROIE using the locally loaded LM Studio model:
```bash
python utils/dataset_benchmark.py \
  --engine local \
  --dataset jsdnrs/ICDAR2019-SROIE \
  --samples 10 \
  --api-base http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b
```

Run the same dataset through Gemini (the key can instead be supplied as `GEMINI_API_KEY`):
```bash
python utils/dataset_benchmark.py \
  --engine gemini \
  --dataset jsdnrs/ICDAR2019-SROIE \
  --samples 10 \
  --api-key "$GEMINI_API_KEY" \
  --model gemini-2.5-flash
```

Live modes deliberately do not fall back to simulated output if the dataset or provider fails.

---

## Output Metrics

The dataset benchmark outputs acceptance rates:
- **AUTO_ACCEPT**: High-confidence extractions safely routed to downstream microservices (~80–90%).
- **NEEDS_REVIEW**: Ambiguous or blurry fields routed to Human-in-the-Loop (HITL) verification (~10%).
- **REJECT**: High-uncertainty hallucinated extractions rejected (~5%).
