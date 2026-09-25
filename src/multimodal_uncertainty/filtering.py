"""Token-likelihood alignment and experimental sample-disagreement utilities."""

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any


def _top_level_value_spans(text: str, field_names: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Find source spans for top-level JSON field values.

    The standard JSON parser does not expose positions. This helper uses the JSON
    decoder to identify each value's exact source span after locating its escaped
    top-level key. It deliberately returns no span for ambiguous or invalid JSON.
    """
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(decoded, dict):
        return {}

    decoder = json.JSONDecoder()
    spans: dict[str, tuple[int, int]] = {}
    for field_name in field_names:
        if field_name not in decoded:
            continue
        key = re.escape(json.dumps(field_name, ensure_ascii=False))
        matches = list(re.finditer(rf"{key}\s*:", text))
        if len(matches) != 1:
            continue
        value_start = matches[0].end()
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        try:
            _, value_end = decoder.raw_decode(text, value_start)
        except json.JSONDecodeError:
            continue
        spans[field_name] = (value_start, value_end)
    return spans


def _locate_content_window(
    token_logprob_pairs: Sequence[tuple[str, float]], generated_text: str
) -> list[tuple[str, float]] | None:
    """Find the token run that reconstructs exactly ``generated_text``.

    Local backends frequently emit hidden reasoning/"thought channel" tokens
    before (and sometimes after) the final answer, so the full logprobs stream
    rarely equals ``message.content`` verbatim anymore. This locates the
    contiguous run of *whole* tokens whose concatenation exactly equals
    ``generated_text`` wherever it falls in the stream, and refuses if that
    text is missing, appears more than once, or is split across a token
    boundary rather than starting/ending cleanly on one.
    """
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token, _ in token_logprob_pairs:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    full_stream = "".join(token for token, _ in token_logprob_pairs)

    start = full_stream.find(generated_text)
    if start == -1 or full_stream.find(generated_text, start + 1) != -1:
        return None
    end = start + len(generated_text)

    start_idx = next((i for i, (s, _) in enumerate(offsets) if s == start), None)
    end_idx = next((i for i, (_, e) in enumerate(offsets) if e == end), None)
    if start_idx is None or end_idx is None:
        return None
    return list(token_logprob_pairs[start_idx : end_idx + 1])


def _token_within_value_span(
    cursor: int, token_end: int, value_start: int, value_end: int, text: str
) -> bool:
    """Check whether a token belongs to a value span, tolerating whitespace overshoot.

    A token that extends past the span boundary only into whitespace (e.g. the
    deterministic separator space in ``": true"``, merged into one BPE token
    with the value itself) still isolates the value's uncertainty almost
    exactly: a certain, invariant separator contributes ~0 to the combined
    log-probability. A token that instead bleeds into a key, quote, comma, or
    brace is genuine structural noise and must still be rejected.
    """
    if token_end <= value_start or cursor >= value_end:
        return False  # no overlap with the value at all
    if cursor < value_start and not text[cursor:value_start].isspace():
        return False  # left overshoot is structural, not just a separator
    if token_end > value_end and not text[value_end:token_end].isspace():
        return False  # right overshoot is structural (comma, brace, quote, ...)
    return True


def isolate_field_value_logprobs(
    token_logprob_pairs: Sequence[tuple[str, float]],
    target_fields: Sequence[str],
    generated_text: str | None = None,
) -> tuple[dict[str, list[float]], list[float]]:
    """Assign token logprobs to JSON value spans rather than token-string heuristics.

    A token is assigned when its character span overlaps a parsed top-level
    JSON value, tolerating whitespace-only overshoot at the boundary (see
    ``_token_within_value_span``). A token that bleeds into actual structural
    syntax (a key, quote, comma, or brace) remains an unavoidable boundary
    approximation; callers should mark such provider output as partial
    evidence if strict field isolation is required.

    Args:
        token_logprob_pairs: Ordered generated token text and chosen-token logprob,
            covering the full generation (any reasoning/channel tokens included).
        target_fields: Top-level schema field names.
        generated_text: Complete assistant JSON text. If omitted, or if it cannot
            be located unambiguously on whole-token boundaries within the token
            stream, no likelihood evidence is returned.

    Returns:
        Per-field likelihoods and their combined list. Empty output means the
        response could not be aligned safely.
    """
    field_logprobs = {field_name: [] for field_name in target_fields}
    if not generated_text:
        return field_logprobs, []

    content_tokens = _locate_content_window(token_logprob_pairs, generated_text)
    if content_tokens is None:
        return field_logprobs, []

    spans = _top_level_value_spans(generated_text, target_fields)
    all_value_logprobs: list[float] = []
    cursor = 0
    for token, logprob in content_tokens:
        token_end = cursor + len(token)
        for field_name, (value_start, value_end) in spans.items():
            if _token_within_value_span(cursor, token_end, value_start, value_end, generated_text):
                field_logprobs[field_name].append(logprob)
                all_value_logprobs.append(logprob)
                break
        cursor = token_end
    return field_logprobs, all_value_logprobs


def calculate_length_normalized_perplexity(logprobs: Sequence[float]) -> float | None:
    """Calculate value-token PPL, returning ``None`` when evidence is absent."""
    if not logprobs:
        return None
    return float(round(math.exp(-sum(logprobs) / len(logprobs)), 3))


def calculate_sample_disagreement_entropy(samples: Sequence[Any]) -> float:
    """Compute lexical entropy over normalized sampled values.

    This is sample disagreement, not semantic entropy: paraphrases are counted as
    distinct values unless a caller clusters them before invoking this function.
    """
    if not samples:
        return 0.0
    counts = Counter(str(sample).strip().lower() for sample in samples)
    total = len(samples)
    return float(
        round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 3)
    )


def calculate_semantic_entropy(samples: Sequence[Any]) -> float:
    """Deprecated compatibility alias for lexical sample-disagreement entropy."""
    return calculate_sample_disagreement_entropy(samples)
