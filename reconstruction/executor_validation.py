"""Auditable comparisons between original and full-path patched execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class FullPathComparison:
	"""Independent pass/fail signals for the standard-path equivalence gate."""

	passed: bool
	generated_tokens_identical: bool
	first_token_logits_close: bool
	maximum_logit_absolute_difference: float
	baseline_layer_order_correct: bool
	patched_layer_order_correct: bool


def compare_full_path_results(
	baseline_sequences: Any,
	patched_sequences: Any,
	baseline_logits: Any,
	patched_logits: Any,
	baseline_layer_order: Sequence[int],
	patched_layer_order: Sequence[int],
	expected_path: Sequence[int],
	torch_module: Any,
	atol: float = 1e-5,
	rtol: float = 1e-4,
) -> FullPathComparison:
	"""Require exact generated tokens, close logits, and exact layer calls."""
	generated_tokens_identical = bool(
		baseline_sequences.shape == patched_sequences.shape
		and torch_module.equal(baseline_sequences, patched_sequences)
	)
	if baseline_logits.shape == patched_logits.shape:
		maximum_difference = float(
			(baseline_logits.float() - patched_logits.float()).abs().max().item(),
		)
		first_token_logits_close = bool(
			torch_module.allclose(
				baseline_logits.float(),
				patched_logits.float(),
				atol=atol,
				rtol=rtol,
			),
		)
	else:
		maximum_difference = float("inf")
		first_token_logits_close = False

	expected = tuple(expected_path)
	baseline_order_correct = tuple(baseline_layer_order) == expected
	patched_order_correct = tuple(patched_layer_order) == expected
	return FullPathComparison(
		passed=(
			generated_tokens_identical
			and first_token_logits_close
			and baseline_order_correct
			and patched_order_correct
		),
		generated_tokens_identical=generated_tokens_identical,
		first_token_logits_close=first_token_logits_close,
		maximum_logit_absolute_difference=maximum_difference,
		baseline_layer_order_correct=baseline_order_correct,
		patched_layer_order_correct=patched_order_correct,
	)
