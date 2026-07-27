"""Pass-at-k metrics for greedy, sampled, and predicted execution programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class TemperatureSelection:
	"""Best sampling accuracy and its temperature for one pass-at-k value."""

	accuracy: float
	temperature: float


def pass_at_k(correctness: Sequence[Sequence[bool]], k: int) -> float:
	"""Return the fraction of questions with a correct answer among the first k."""
	if k <= 0:
		raise ValueError("k must be positive")
	if not correctness:
		return 0.0
	for question_index, samples in enumerate(correctness):
		if len(samples) < k:
			raise ValueError(
				f"Question {question_index} has {len(samples)} samples, fewer than k={k}",
			)
	correct_questions = sum(any(samples[:k]) for samples in correctness)
	return correct_questions / len(correctness)


def pass_at_k_curve(
	correctness: Sequence[Sequence[bool]],
	maximum_k: int = 5,
) -> Dict[int, float]:
	"""Compute the nested empirical pass-at-1 through pass-at-maximum-k curve."""
	if maximum_k <= 0:
		raise ValueError("maximum_k must be positive")
	return {k: pass_at_k(correctness, k) for k in range(1, maximum_k + 1)}


def best_sampling_temperature_curve(
	correctness_by_temperature: Mapping[float, Sequence[Sequence[bool]]],
	maximum_k: int = 5,
) -> Dict[int, TemperatureSelection]:
	"""Select the best reported sampling temperature independently for each k."""
	if not correctness_by_temperature:
		raise ValueError("At least one sampling temperature is required")
	curves = {
		float(temperature): pass_at_k_curve(correctness, maximum_k)
		for temperature, correctness in correctness_by_temperature.items()
	}
	selected = {}
	for k in range(1, maximum_k + 1):
		temperature, accuracy = max(
			(
				(temperature, curve[k])
				for temperature, curve in curves.items()
			),
			key=lambda item: (item[1], -item[0]),
		)
		selected[k] = TemperatureSelection(
			accuracy=accuracy,
			temperature=temperature,
		)
	return selected
