"""Evidence-aware routing for experimentally calibrated extraction signals."""

from collections.abc import Sequence

from multimodal_uncertainty.models import E2EEvaluatorResult, EvidenceStatus, NodeUncertainty


class PipelineDecisionEngine:
    """Route only calibrated, available evidence; otherwise preserve uncertainty."""

    def __init__(
        self,
        ppl_threshold: float = 2.2,
        entropy_threshold: float = 0.5,
        reject_ppl_multiplier: float = 2.0,
        reject_entropy_threshold: float = 1.2,
        calibration_id: str | None = None,
    ) -> None:
        self.ppl_threshold = ppl_threshold
        self.entropy_threshold = entropy_threshold
        self.reject_ppl_threshold = ppl_threshold * reject_ppl_multiplier
        self.reject_entropy_threshold = reject_entropy_threshold
        self.calibration_id = calibration_id

    def evaluate(self, node_results: Sequence[NodeUncertainty]) -> E2EEvaluatorResult:
        """Evaluate node evidence using field bottlenecks and task calibration identity."""
        nodes = list(node_results)
        if not nodes:
            return E2EEvaluatorResult(action="INSUFFICIENT_EVIDENCE", node_results=[])

        statuses = {node.evidence_status for node in nodes}
        if EvidenceStatus.PROVIDER_ERROR in statuses:
            return E2EEvaluatorResult(action="EXECUTION_FAILED", node_results=nodes)
        if EvidenceStatus.INVALID_OUTPUT in statuses:
            return E2EEvaluatorResult(action="REJECT", node_results=nodes)
        if statuses != {EvidenceStatus.AVAILABLE}:
            return E2EEvaluatorResult(action="INSUFFICIENT_EVIDENCE", node_results=nodes)
        if self.calibration_id and any(
            node.calibration_id != self.calibration_id for node in nodes
        ):
            return E2EEvaluatorResult(action="INSUFFICIENT_EVIDENCE", node_results=nodes)

        field_ppls = [
            field.perplexity
            for node in nodes
            for field in node.fields
            if field.perplexity is not None
        ]
        node_ppls = [
            node.overall_value_perplexity for node in nodes if node.overall_value_perplexity
        ]
        bottleneck = max([*field_ppls, *node_ppls], default=None)
        mean_ppl = sum(node_ppls) / len(node_ppls) if node_ppls else None
        max_entropy = max(
            (field.sample_disagreement_entropy for node in nodes for field in node.fields),
            default=0.0,
        )

        if bottleneck is None:
            action = "INSUFFICIENT_EVIDENCE"
        elif bottleneck > self.reject_ppl_threshold or max_entropy > self.reject_entropy_threshold:
            action = "REJECT"
        elif bottleneck > self.ppl_threshold or max_entropy > self.entropy_threshold:
            action = "NEEDS_REVIEW"
        else:
            action = "AUTO_ACCEPT"

        return E2EEvaluatorResult(
            action=action,
            bottleneck_perplexity=round(bottleneck, 3) if bottleneck is not None else None,
            mean_e2e_perplexity=round(mean_ppl, 3) if mean_ppl is not None else None,
            max_field_entropy=round(max_entropy, 3),
            node_results=nodes,
        )
