# Architectural Intent (`INTENT.md`)

## Core Philosophy

Traditional LLM evaluation relies heavily on exact sequence perplexity or raw candidate log-probabilities. However, in structured multimodal extractions (PDFs, receipts, medical records, invoices):

1. **Structural JSON Syntax is Noise**: Tokens like `{`, `"key":`, `}`, `\n` are deterministic and generated with near-100% confidence by fine-tuned or instruction-tuned models. Including these in perplexity calculations artificially suppresses actual field extraction uncertainty.
2. **Values Are What Matter**: Only the token probabilities corresponding to extracted value contents reflect genuine model confidence or confusion.
3. **Semantic Paraphrasing vs. Direct Contradiction**: Exact token matching fails to differentiate formatting variations (`$10.00` vs `10.00`) from actual semantic hallucinations (`cat` vs `dog`). Semantic entropy sampling measures conceptual distribution divergence.

## Mathematical Formulation

For a target field $f$ with token sequence $V_f = (t_1, t_2, \dots, t_N)$ representing extracted values:

### Value-Only Length-Normalized Perplexity ($PPL$)

$$\text{LogProb}(V_f) = \sum_{i=1}^{N} \log P(t_i \mid t_{<i}, \text{Prompt}, \text{Image})$$

$$PPL(f) = \exp\left(-\frac{1}{N} \text{LogProb}(V_f)\right)$$

### Semantic Entropy ($SE$)

For $M$ stochastic samples under temperature $T > 0$, grouped into semantic equivalence classes $C_1, C_2, \dots, C_k$:

$$SE(f) = -\sum_{j=1}^{k} p(C_j) \log_2 p(C_j)$$

where $p(C_j) = \frac{|C_j|}{M}$.

---

## Decision Routing Boundaries

- **AUTO_ACCEPT**: $PPL(f) \le 2.2$ and $SE(f) \le 0.5$ for all fields.
- **NEEDS_REVIEW**: $PPL(f) > 2.2$ or $SE(f) > 0.5$ for any field.
- **REJECT**: $PPL(f) > 4.4$ or $SE(f) > 1.2$ for any bottleneck field.
