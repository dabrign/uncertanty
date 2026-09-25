"""
Multi-Step Agent Pipeline & Chain of Prompts Uncertainty Tracker.

Implements stepwise perplexity tracking across sequential agent nodes (Cui et al., 2025)
and enforces upstream uncertainty tag propagation to prevent cascading failures.
"""

from multimodal_uncertainty.decision import PipelineDecisionEngine
from multimodal_uncertainty.models import E2EEvaluatorResult, NodeUncertainty


class ChainOfPromptsEvaluator:
    """
    Evaluates step-by-step intermediate perplexities in multi-stage agent workflows.

    Example Workflow:
    Step 1: Document Classification & Layout Routing -> NodeUncertainty 1
    Step 2: Key Information Extraction               -> NodeUncertainty 2
    Step 3: Validation & Business Rules               -> NodeUncertainty 3
    """

    def __init__(self, decision_engine: PipelineDecisionEngine | None = None):
        self.decision_engine = decision_engine or PipelineDecisionEngine()
        self.pipeline_nodes: list[NodeUncertainty] = []

    def add_step_result(self, node_result: NodeUncertainty) -> str:
        """
        Appends a node step result to the pipeline execution trace.
        Returns the action ('AUTO_ACCEPT', 'NEEDS_REVIEW', 'REJECT') for this node.
        """
        self.pipeline_nodes.append(node_result)
        eval_res = self.decision_engine.evaluate([node_result])
        return eval_res.action

    def get_upstream_warning_context(self) -> str:
        """
        Returns warning context text to append to downstream agent prompts if
        upstream nodes exhibited high uncertainty.
        """
        eval_res = self.decision_engine.evaluate(self.pipeline_nodes)
        if eval_res.action in ("NEEDS_REVIEW", "REJECT"):
            return (
                f"\n[UPSTREAM UNCERTAINTY WARNING]: Upstream agent step exhibited "
                f"high perplexity ({eval_res.bottleneck_perplexity}). Re-verify fields carefully.\n"
            )
        return ""

    def evaluate_end_to_end(self) -> E2EEvaluatorResult:
        """
        Computes End-to-End pipeline metrics across all executed steps.
        Identifies the exact bottleneck step causing pipeline degradation.
        """
        return self.decision_engine.evaluate(self.pipeline_nodes)

    def identify_bottleneck_step(self) -> tuple[str | None, float]:
        """
        Finds the specific node step with the highest overall value perplexity.
        Returns (node_name, max_perplexity).
        """
        if not self.pipeline_nodes:
            return None, 1.0

        bottleneck = max(self.pipeline_nodes, key=lambda n: n.overall_value_perplexity)
        return bottleneck.node_name, bottleneck.overall_value_perplexity
