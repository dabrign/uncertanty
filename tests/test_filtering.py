"""Tests for tokenizer-safe JSON value likelihood alignment and sample disagreement."""

import math

from multimodal_uncertainty.filtering import (
    calculate_length_normalized_perplexity,
    calculate_sample_disagreement_entropy,
    isolate_field_value_logprobs,
)


def test_isolate_field_value_logprobs_aligns_json_value_spans() -> None:
    text = '{"invoice_number": "INV-1092", "amount": 250.00}'
    token_pairs = [
        ('{"invoice', 0.0),
        ('_number": "', 0.0),
        ("INV-1092", -0.05),
        ('", "amount": ', 0.0),
        ("250.00", -0.10),
        ("}", 0.0),
    ]

    field_logprobs, all_value_logprobs = isolate_field_value_logprobs(
        token_pairs, ["invoice_number", "amount"], text
    )

    assert field_logprobs["invoice_number"] == [-0.05]
    assert field_logprobs["amount"] == [-0.10]
    assert all_value_logprobs == [-0.05, -0.10]


def test_isolation_refuses_unaligned_token_stream() -> None:
    field_logprobs, all_value_logprobs = isolate_field_value_logprobs(
        [("different", -0.1)], ["answer"], '{"answer": "ok"}'
    )
    assert field_logprobs == {"answer": []}
    assert all_value_logprobs == []


def test_isolate_field_value_logprobs_skips_leading_reasoning_channel_tokens() -> None:
    # Regression for local backends (e.g. Ollama) that stream a hidden
    # reasoning/thought-channel preamble ahead of the final JSON answer, so
    # the logprobs stream is longer than message.content but still contains
    # it verbatim on clean token boundaries.
    generated_text = '{"item_name": "sofa", "price": 450.0}'
    token_pairs = [
        ("<|channel|>", -0.01),
        ("thought", -0.02),
        ("\n", -0.01),
        ("...reasoning about the request...", -0.03),
        ('{"item_name": "', 0.0),
        ("sofa", -0.05),
        ('", "price": ', 0.0),
        ("450.0", -0.08),
        ("}", 0.0),
    ]

    field_logprobs, all_value_logprobs = isolate_field_value_logprobs(
        token_pairs, ["item_name", "price"], generated_text
    )

    assert field_logprobs["item_name"] == [-0.05]
    assert field_logprobs["price"] == [-0.08]
    assert all_value_logprobs == [-0.05, -0.08]


def test_isolate_field_value_logprobs_tolerates_whitespace_merged_boolean_token() -> None:
    # Regression for a real Ollama/gemma4:12b capture: the tokenizer merges the
    # deterministic ": " separator with a short boolean literal into one token
    # (" true"), so the token's character span starts one character before the
    # value's own span. That leading overshoot is pure whitespace, not
    # structural syntax, so it should still count as value-only evidence.
    generated_text = '{"is_present": true, "note": "ok"}'
    token_pairs = [
        ('{"is_present":', 0.0),
        (" true", -1.1920928955078125e-07),
        (', "note": ', 0.0),
        ('"ok"', -0.03),
        ("}", 0.0),
    ]

    field_logprobs, all_value_logprobs = isolate_field_value_logprobs(
        token_pairs, ["is_present", "note"], generated_text
    )

    assert field_logprobs["is_present"] == [-1.1920928955078125e-07]
    assert field_logprobs["note"] == [-0.03]
    assert all_value_logprobs == [-1.1920928955078125e-07, -0.03]


def test_isolate_field_value_logprobs_still_rejects_structural_overshoot() -> None:
    # A token that bleeds into a real structural character (here, the comma
    # right after the value) must still be excluded even under the new
    # whitespace tolerance -- only whitespace overshoot is forgiven.
    generated_text = '{"flag": true,"note": "ok"}'
    token_pairs = [
        ('{"flag":', 0.0),
        (" true,", -0.02),  # overshoots into the comma, not just whitespace
        ('"note": ', 0.0),
        ('"ok"', -0.03),
        ("}", 0.0),
    ]

    field_logprobs, all_value_logprobs = isolate_field_value_logprobs(
        token_pairs, ["flag", "note"], generated_text
    )

    assert field_logprobs["flag"] == []
    assert field_logprobs["note"] == [-0.03]
    assert all_value_logprobs == [-0.03]


def test_isolation_refuses_when_content_splits_a_token_boundary() -> None:
    # If the reasoning preamble bleeds into the same token as the JSON start,
    # we cannot cleanly attribute logprobs to the value span, so no evidence
    # should be returned rather than a misaligned guess.
    generated_text = '{"answer": "ok"}'
    token_pairs = [
        ("prefix", -0.01),
        ('text{"answer": "ok"}', -0.02),
    ]

    field_logprobs, all_value_logprobs = isolate_field_value_logprobs(
        token_pairs, ["answer"], generated_text
    )

    assert field_logprobs == {"answer": []}
    assert all_value_logprobs == []


def test_calculate_length_normalized_perplexity() -> None:
    assert calculate_length_normalized_perplexity([0.0, 0.0]) == 1.0
    assert math.isclose(calculate_length_normalized_perplexity([-0.693147]) or 0, 2.0, rel_tol=1e-2)
    assert calculate_length_normalized_perplexity([]) is None


def test_calculate_sample_disagreement_entropy() -> None:
    assert calculate_sample_disagreement_entropy(["cat", "cat", "cat"]) == 0.0
    assert calculate_sample_disagreement_entropy(["cat", "dog"]) == 1.0
    assert calculate_sample_disagreement_entropy([]) == 0.0
