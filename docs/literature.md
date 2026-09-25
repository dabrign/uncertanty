# Literature Review: Value-Only Logprob Filtering for Uncertainty Quantification

This document expands on the [Scientific Foundations](../README.md#scientific-foundations) section of the
README. It covers the three papers this project's methods are directly built on, plus additional literature
surveyed to identify known flaws or possible improvements to the overall approach: computing perplexity and
entropy only over JSON *value* tokens, isolated by character span, from an instruction-tuned model's raw
output logprobs.

## Summary Table

| Paper | Year | Claim used in this project | Where it's implemented | Known limitation for this project |
|---|---|---|---|---|
| Kuhn, Tran, Gal — *Semantic Uncertainty* | 2023 | Cluster generations into semantic equivalence classes before computing entropy, rather than exact-string matching | `calculate_semantic_entropy` (alias) | Not actually implemented — see below |
| Malinin & Gales — *Uncertainty Estimation in Autoregressive Structured Prediction* | 2020 | Length-normalized perplexity: $PPL(f) = \exp(-\frac{1}{N}\sum \log P(t_i))$ | `calculate_length_normalized_perplexity` | Paper's main contribution is aleatoric/epistemic decomposition via ensembles; this project has no ensemble, only single-model repeated sampling |
| Cui et al. — stepwise perplexity for agentic pipelines | 2025 | Isolate intermediate token log-probabilities to pinpoint bottleneck steps in multi-step chains | `ChainOfPromptsEvaluator` (`pipeline.py`) | N/A — used as intended |
| Tian, Mitchell, et al. — *Just Ask for Calibration* | 2023 | Verbalized confidence is better calibrated than raw token logprobs for RLHF/instruction-tuned models | Not implemented (found during literature search) | Direct calibration ceiling: this project extracts raw logprobs from instruction-tuned chat models (e.g. gemma via Ollama) exactly where this paper shows that signal is weaker than an alternative |
| Kossen, Han, et al. — *Semantic Entropy Probes* | 2024 | Approximate semantic entropy from a single forward pass's hidden states, avoiding multi-sample cost | Not implemented (found during literature search) | Improvement candidate: could replace/supplement `entropy_samples`-based sampling, which pays for N full generations per field |
| Farquhar, Kossen, Kuhn, Gal — *Detecting Hallucinations in LLMs Using Semantic Entropy* | 2024 (Nature) | Large-scale validation of NLI-entailment clustering for hallucination detection | Not implemented (found during literature search) | Entailment clustering fits free-text answers, not atomic values (booleans, numbers) — validates why this project's entropy function is lexical, not semantic |
| Kadavath et al. — *Language Models (Mostly) Know What They Know* | 2022 | Models can self-evaluate P(True)/P(IK) as an uncertainty signal | Not implemented (found during literature search) | Calibration of P(IK) degrades on unfamiliar/out-of-distribution tasks — a caveat for generalizing self-evaluation to new extraction domains |
| Abbasi Yadkori et al. — *To Believe or Not to Believe Your LLM* | 2024 | Iterative-prompting-based epistemic/aleatoric uncertainty detection without needing logprobs at all | Not implemented (found during literature search) | Candidate alternative architecture if logprob access is ever unavailable (e.g. some hosted APIs) |
| Microsoft `guidance` — "token healing" | — | Greedy BPE tokenization creates boundary artifacts between deterministic and generated text | Directly relevant prior art for the fix in `filtering.py`'s `_token_within_value_span` | Confirms the whitespace-tolerance fix matches standard practice, not a project-specific hack |

## The three cited foundations

### 1. Semantic Uncertainty (Kuhn et al., 2023)

Proposes measuring unpredictability by clustering multiple sampled generations into semantic equivalence
classes — via NLI entailment — before computing Shannon entropy over the cluster proportions, rather than
treating every distinct string as a separate outcome.

**What this project actually implements:** `calculate_sample_disagreement_entropy` computes entropy over
*exact-string* matches (case/whitespace-normalized), not semantic clusters. Its own docstring says so
explicitly: *"this is sample disagreement, not semantic entropy: paraphrases are counted as distinct values
unless a caller clusters them."* `calculate_semantic_entropy` is a deprecated alias for the same lexical
function — no NLI clustering step exists in the codebase.

**Why that's defensible, not just a shortcut:** the Farquhar et al. 2024 follow-up (below) uses this same
entailment-clustering machinery for hallucination detection over free-text answers, where "meaning-equivalent
paraphrase" is a meaningful category. For a schema field like `is_present: bool` or `price: float`, entailment
between two literal values is degenerate — there is no paraphrase axis to cluster over. Lexical entropy is
the *correct* choice for atomic fields; it would only under-count disagreement for long free-text fields
(e.g. `confidence_reasoning`) where two paraphrased explanations should arguably count as agreement.

### 2. Length-Normalized Perplexity (Malinin & Gales, 2020)

$$PPL(f) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(t_i \mid t_{<i})\right)$$

This formula is correctly implemented in `calculate_length_normalized_perplexity`, and is exactly what this
project needs to avoid penalizing long field values relative to short ones.

**The gap:** length-normalized PPL is a small piece of this paper. Its central contribution is decomposing
total predictive uncertainty into **aleatoric** (irreducible, e.g. genuinely ambiguous input) and
**epistemic** (reducible, e.g. the model doesn't know this domain) components, using disagreement across an
*ensemble* of models. This project has no ensemble — `overall_value_perplexity` is a single number from a
single model, and `calculate_sample_disagreement_entropy` only captures spread across repeated samples of
that *same* model. It is a cheap proxy for epistemic uncertainty, not the ensemble-based decomposition the
cited paper actually proposes. In practice this means the pipeline cannot currently distinguish "this field
is inherently unclear from the image" from "this model doesn't know this concept."

### 3. Stepwise Perplexity Reasoning (Cui et al., 2025)

Used as intended in `ChainOfPromptsEvaluator` (`pipeline.py`) to isolate per-step log-probabilities in a
multi-step/agentic chain, so a bottleneck step can be pinpointed rather than only the end-to-end confidence.
No gap identified here relative to how the project uses it.

## Additional literature found (flaws and possible improvements)

### The tokenizer-boundary problem is known, cross-project prior art

The bug fixed this session — a BPE tokenizer merging the deterministic `": "` separator into the same token
as a short value like `true` — is the same root cause addressed by **"token healing"** in Microsoft's
`guidance` library (github.com/guidance-ai/guidance): greedy tokenization does not respect the boundaries a
downstream consumer (a prompt template, or here, a JSON value span) actually cares about. This is confirmation
that the whitespace-tolerant containment check in `_token_within_value_span` is the standard fix pattern for
this class of problem, not a one-off patch.

### A calibration ceiling this project inherits: raw logprobs vs. verbalized confidence

Tian et al., 2023, *"Just Ask for Calibration"* (arXiv:2305.14975), found that for RLHF/instruction-tuned
models (ChatGPT, GPT-4, Claude), **verbalized confidence** — asking the model to state a confidence score in
text — is typically *better calibrated* than the model's raw token-level conditional probabilities, reducing
expected calibration error by roughly 50% relative in their benchmarks.

This matters directly here: `LocalOpenAIUncertaintyEngine` extracts raw `logprobs.content` from exactly the
kind of model this finding is about — an instruction-tuned chat model (e.g. `gemma4:12b` served via Ollama).
The whitespace-tolerance fix improves *attribution accuracy* (which tokens the logprob evidence comes from),
but it cannot improve the underlying *calibration* of that evidence. This is a ceiling on the value-only-logprob
strategy as a category, independent of any implementation bug.

### A cost-reduction opportunity: Semantic Entropy Probes

Kossen et al., 2024, *"Semantic Entropy Probes"* (arXiv:2406.15927), approximate semantic entropy from a
single forward pass's hidden states, avoiding the need for multiple sampled generations that full semantic
entropy (and this project's `entropy_samples`-driven sampling) requires. If sampling cost ever becomes a
concern, this is the literature's direct answer — though it would require access to hidden states, which the
OpenAI-compatible HTTP API surface this project uses does not expose.

### Validates the project's self-scoped-down entropy function

Farquhar, Kossen, Kuhn, Gal, 2024 (*Nature*), the large-scale hallucination-detection extension of the
already-cited Kuhn et al. 2023, clusters sampled answers via NLI entailment. As discussed above, that
clustering step is well-suited to free-text answers and a poor fit for atomic scalar values — which is exactly
why this project's own entropy function documents itself as lexical rather than semantic. The literature
confirms this is the right scope, not a missing feature.

### Other relevant signals, lower priority

- Kadavath et al., 2022, *"Language Models (Mostly) Know What They Know"* (arXiv:2207.05221): models can
  self-report P(True) confidence reasonably well, but P(IK) ("do I know this?") calibration degrades on
  unfamiliar tasks — a caveat if this approach is ever extended to self-evaluation prompts.
- Abbasi Yadkori et al., 2024, *"To Believe or Not to Believe Your LLM"* (arXiv:2406.02543): detects
  hallucinations via iterative prompting and an information-theoretic epistemic/aleatoric split, without
  requiring logprob access at all — a candidate alternative architecture for providers that don't expose
  token logprobs (many hosted APIs don't).

## Net assessment

None of the additional literature contradicts the project's cited foundations or the whitespace-tolerance fix.
What it adds is a ceiling: value-only logprob filtering, done correctly, still inherits (1) whatever
calibration gap exists between an instruction-tuned model's raw token probabilities and its "true" confidence,
and (2) the absence of an aleatoric/epistemic decomposition, since there is no ensemble. Both are addressable
in future work (verbalized-confidence elicitation; semantic entropy probes) without changing the current
architecture's core principle that structural JSON syntax is noise and only value tokens carry signal.
