"""
Demo Benchmark Script: Multi-Step Agent Pipeline & Chain of Prompts Uncertainty Tracking.

Simulates a 3-step document extraction pipeline:
Step 1: Document Classification (Low PPL)
Step 2: Field Extraction (High PPL Bottleneck on ambiguous date field)
Step 3: Upstream Warning Injection & Downstream Re-verification
"""

from multimodal_uncertainty.models import FieldUncertainty, NodeUncertainty
from multimodal_uncertainty.pipeline import ChainOfPromptsEvaluator


def run_chain_of_prompts_demo():
    print("==========================================================================")
    print("  RUNNING MULTI-STEP AGENT PIPELINE & CHAIN OF PROMPTS UQ DEMO")
    print("==========================================================================")

    pipeline = ChainOfPromptsEvaluator()

    # Step 1: Document Classification & Layout Routing
    step1_node = NodeUncertainty(
        node_name="Step1_Doc_Classification_Agent",
        overall_value_perplexity=1.04,
        fields=[
            FieldUncertainty(field_name="doc_type", extracted_value="TAX_INVOICE", perplexity=1.02, semantic_entropy=0.0),
            FieldUncertainty(field_name="language", extracted_value="en-US", perplexity=1.06, semantic_entropy=0.0),
        ]
    )
    action1 = pipeline.add_step_result(step1_node)
    print(f"\n[Step 1 Execution]: {step1_node.node_name}")
    print(f"  PPL: {step1_node.overall_value_perplexity} | Action: {action1}")

    # Step 2: Field Extraction (Contains ambiguous blurred date)
    step2_node = NodeUncertainty(
        node_name="Step2_Field_Extraction_Agent",
        overall_value_perplexity=3.85,  # Bottleneck!
        fields=[
            FieldUncertainty(field_name="invoice_id", extracted_value="INV-9921", perplexity=1.08, semantic_entropy=0.0),
            FieldUncertainty(field_name="total_amount", extracted_value=1450.00, perplexity=1.12, semantic_entropy=0.0),
            FieldUncertainty(field_name="issue_date", extracted_value="2026-10-15?", perplexity=4.95, semantic_entropy=1.10),
        ]
    )
    action2 = pipeline.add_step_result(step2_node)
    print(f"\n[Step 2 Execution]: {step2_node.node_name}")
    print(f"  PPL: {step2_node.overall_value_perplexity} | Action: {action2}")

    # Check upstream warning tag generated for Step 3 prompt
    warning_tag = pipeline.get_upstream_warning_context()
    print(f"\n[Propagating Warning to Step 3 Context Prompt]:\n{warning_tag.strip()}")

    # Step 3: Validation & Downstream Spatial Re-verification
    step3_node = NodeUncertainty(
        node_name="Step3_Validation_ReVerification_Agent",
        overall_value_perplexity=1.10,
        fields=[
            FieldUncertainty(field_name="issue_date_verified", extracted_value="2026-10-15", perplexity=1.10, semantic_entropy=0.0),
            FieldUncertainty(field_name="validation_status", extracted_value="PASSED", perplexity=1.05, semantic_entropy=0.0),
        ]
    )
    action3 = pipeline.add_step_result(step3_node)
    print(f"\n[Step 3 Execution]: {step3_node.node_name}")
    print(f"  PPL: {step3_node.overall_value_perplexity} | Action: {action3}")

    # End-to-End Evaluation
    e2e_res = pipeline.evaluate_end_to_end()
    bottleneck_node, bottleneck_ppl = pipeline.identify_bottleneck_step()

    print("\n----------------------------------------------------------------------")
    print("  END-TO-END PIPELINE EVALUATION SUMMARY")
    print("----------------------------------------------------------------------")
    print(f"End-to-End Action:        {e2e_res.action}")
    print(f"Mean E2E Perplexity:      {e2e_res.mean_e2e_perplexity}")
    print(f"Bottleneck Perplexity:    {e2e_res.bottleneck_perplexity}")
    print(f"Pipeline Bottleneck Step: {bottleneck_node} (PPL: {bottleneck_ppl})")
    print("==========================================================================")


if __name__ == "__main__":
    run_chain_of_prompts_demo()
