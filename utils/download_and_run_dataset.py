"""
Dataset Downloader & Evaluation Runner Script.

Downloads public datasets from Hugging Face Hub (SROIE receipts, CORD),
evaluates extraction perplexity and semantic entropy, and exports results to JSON/CSV
ready for uploading into the React HITL Visualizer Dashboard.
"""

import argparse
import base64
import csv
import json
import os
from io import BytesIO
from pydantic import BaseModel, Field

from multimodal_uncertainty.decision import PipelineDecisionEngine
from multimodal_uncertainty.engines.local_openai import LocalOpenAIUncertaintyEngine

try:
    from datasets import load_dataset
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False


class SROIEReceiptSchema(BaseModel):
    company: str = Field(description="Company or store name")
    date: str = Field(description="Transaction date")
    address: str = Field(description="Store address")
    total: float = Field(description="Total transaction amount")


def download_and_evaluate(
    dataset_name: str = "jsdnrs/ICDAR2019-SROIE",
    split: str = "test",
    num_samples: int = 5,
    output_json: str = "dataset_export.json",
    output_csv: str = "dataset_export.csv",
    api_base: str = "http://localhost:11434/v1",
    model: str = "qwen2.5-vl",
    mock: bool = False,
):
    """Evaluate live dataset samples, or print explicitly-requested synthetic ones.

    Live mode never falls back to synthetic data: a missing `datasets` install,
    a dataset load failure, or an extraction error aborts the run instead of
    silently substituting fabricated records into the export files.
    """
    print("==========================================================================")
    print(f"  DOWNLOADING & EVALUATING DATASET: {dataset_name} ({split} split)")
    print("==========================================================================")

    if mock:
        print("MOCK MODE: generating synthetic export records, no dataset or model calls made.")
        export_records = generate_synthetic_dataset_records(num_samples)
    else:
        if not HAS_HUGGINGFACE:
            raise ImportError(
                "Live dataset evaluation requires the `datasets` package. Install with "
                "`pip install datasets`, or pass --mock to generate synthetic records instead."
            )

        print(f"Downloading dataset `{dataset_name}` from Hugging Face...")
        ds = load_dataset(dataset_name, split=split)
        engine = LocalOpenAIUncertaintyEngine(api_base=api_base, model=model)
        decision_engine = PipelineDecisionEngine(ppl_threshold=2.2, entropy_threshold=0.5)

        export_records = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break

            img = item.get("image")
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            node_res = engine.extract_and_evaluate(
                prompt="Extract company, date, address, and total amount from this receipt.",
                schema_cls=SROIEReceiptSchema,
                image_base64=img_b64,
                entropy_samples=2,
            )
            eval_res = decision_engine.evaluate([node_res])

            fields_data = [f.model_dump() for f in node_res.fields]
            record = {
                "sample_id": f"SROIE_{i+1}",
                "node_name": node_res.node_name,
                "action": eval_res.action,
                "bottleneck_perplexity": eval_res.bottleneck_perplexity,
                "mean_e2e_perplexity": eval_res.mean_e2e_perplexity,
                "max_field_entropy": eval_res.max_field_entropy,
                "is_synthetic": False,
                "fields": fields_data,
            }
            export_records.append(record)
            print(f"Sample #{i+1}: Action={eval_res.action}, Bottleneck PPL={eval_res.bottleneck_perplexity}")

        if not export_records:
            raise ValueError(f"Dataset `{dataset_name}` split `{split}` did not contain any samples.")

    # 1. Save JSON
    with open(output_json, "w") as f:
        json.dump(export_records, f, indent=2)
    print(f"\n✅ Exported JSON dataset evaluation report: {output_json}")

    # 2. Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "node_name", "action", "bottleneck_perplexity",
            "mean_e2e_perplexity", "max_field_entropy", "is_synthetic", "field_name",
            "extracted_value", "perplexity", "semantic_entropy"
        ])
        for rec in export_records:
            for field in rec.get("fields", []):
                writer.writerow([
                    rec["sample_id"], rec["node_name"], rec["action"],
                    rec["bottleneck_perplexity"], rec["mean_e2e_perplexity"],
                    rec["max_field_entropy"], rec["is_synthetic"], field["field_name"],
                    field["extracted_value"], field["perplexity"], field["semantic_entropy"]
                ])
    print(f"✅ Exported CSV dataset evaluation report:  {output_csv}")

    print("\n----------------------------------------------------------------------")
    print("  HOW TO VISUALIZE RESULTS IN THE REACT DASHBOARD:")
    print("  1. Open website/dashboard.html in your browser or run:")
    print("     python3 -m http.server 8000 --directory website")
    print("  2. Open http://localhost:8000/dashboard.html")
    print(f"  3. Drag and drop `{output_json}` or `{output_csv}` to view interactive field highlighting!")
    print("==========================================================================")


def generate_synthetic_dataset_records(num_samples: int = 5):
    """Generates synthetic benchmark evaluation records for testing the React dashboard.

    Every record is tagged `is_synthetic: True` so exported JSON/CSV consumers
    (including the dashboard) can never mistake this for live model output.
    """
    synthetic = [
        {
            "sample_id": "REC_001_CLEAN",
            "node_name": "Receipt_Extraction_Node",
            "action": "AUTO_ACCEPT",
            "bottleneck_perplexity": 1.05,
            "mean_e2e_perplexity": 1.04,
            "max_field_entropy": 0.0,
            "is_synthetic": True,
            "fields": [
                {"field_name": "company", "extracted_value": "SUPERMARKET METRO", "perplexity": 1.02, "semantic_entropy": 0.0},
                {"field_name": "date", "extracted_value": "2026-08-14", "perplexity": 1.05, "semantic_entropy": 0.0},
                {"field_name": "address", "extracted_value": "VIA ROMA 45, MILANO", "perplexity": 1.04, "semantic_entropy": 0.0},
                {"field_name": "total", "extracted_value": 45.90, "perplexity": 1.03, "semantic_entropy": 0.0},
            ],
        },
        {
            "sample_id": "REC_002_BLURRY_DATE",
            "node_name": "Receipt_Extraction_Node",
            "action": "NEEDS_REVIEW",
            "bottleneck_perplexity": 2.85,
            "mean_e2e_perplexity": 1.48,
            "max_field_entropy": 0.65,
            "is_synthetic": True,
            "fields": [
                {"field_name": "company", "extracted_value": "CAFE CENTRAL", "perplexity": 1.08, "semantic_entropy": 0.0},
                {"field_name": "date", "extracted_value": "2026-??-12", "perplexity": 2.85, "semantic_entropy": 0.65},
                {"field_name": "address", "extracted_value": "MAIN STREET 12", "perplexity": 1.10, "semantic_entropy": 0.0},
                {"field_name": "total", "extracted_value": 12.50, "perplexity": 1.06, "semantic_entropy": 0.0},
            ],
        },
        {
            "sample_id": "REC_003_HALLUCINATED_TOTAL",
            "node_name": "Receipt_Extraction_Node",
            "action": "REJECT",
            "bottleneck_perplexity": 5.40,
            "mean_e2e_perplexity": 2.25,
            "max_field_entropy": 1.35,
            "is_synthetic": True,
            "fields": [
                {"field_name": "company", "extracted_value": "PHARMACY PLUS", "perplexity": 1.05, "semantic_entropy": 0.0},
                {"field_name": "date", "extracted_value": "2026-09-01", "perplexity": 1.08, "semantic_entropy": 0.0},
                {"field_name": "address", "extracted_value": "7TH AVENUE", "perplexity": 1.12, "semantic_entropy": 0.0},
                {"field_name": "total", "extracted_value": 9999.99, "perplexity": 5.40, "semantic_entropy": 1.35},
            ],
        },
    ]
    return synthetic[:num_samples]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Downloader & Evaluator")
    parser.add_argument("--dataset", default="jsdnrs/ICDAR2019-SROIE", help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--output-json", default="dataset_export.json", help="JSON output file")
    parser.add_argument("--output-csv", default="dataset_export.csv", help="CSV output file")
    parser.add_argument("--api-base", default="http://localhost:11434/v1", help="Local OpenAI-compatible endpoint")
    parser.add_argument("--model", default="qwen2.5-vl", help="Local model identifier")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic export records instead of calling a live dataset/model.",
    )
    args = parser.parse_args()

    download_and_evaluate(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.samples,
        output_json=args.output_json,
        output_csv=args.output_csv,
        api_base=args.api_base,
        model=args.model,
        mock=args.mock,
    )
