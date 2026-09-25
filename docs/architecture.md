# Architecture & Scientific Foundations

## Why Value-Only Logprob Filtering Matters

Traditional LLM evaluation strategies calculate output token perplexity across the entire generated JSON string. However:

1. **Structural JSON Syntax is Noise**: Fixed tokens like `{`, `"key":`, `}`, `\n` are deterministic and generated with near-100% confidence by fine-tuned or instruction-tuned models.
2. **Value Tokens Contain Risk**: Only the token probabilities corresponding to extracted value contents reflect genuine model confidence or confusion.

---

## Mathematical Formulation

For a target field $f$ with value token sequence $V_f = (t_1, t_2, \dots, t_N)$:

### Value-Only Length-Normalized Perplexity ($PPL$)

$$\text{LogProb}(V_f) = \sum_{i=1}^{N} \log P(t_i \mid t_{<i}, \text{Prompt}, \text{Image})$$

$$PPL(f) = \exp\left(-\frac{1}{N} \text{LogProb}(V_f)\right)$$

### Semantic Entropy ($SE$)

For $M$ stochastic samples under temperature $T > 0$, grouped into semantic equivalence classes $C_1, C_2, \dots, C_k$:

$$SE(f) = -\sum_{j=1}^{k} p(C_j) \log_2 p(C_j)$$

where $p(C_j) = \frac{|C_j|}{M}$.
