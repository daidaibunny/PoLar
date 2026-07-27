"""Select a high-throughput CUDA batch with explicit memory headroom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BatchMeasurement:
	"""One measured greedy-generation batch result."""

	batch_size: int
	succeeded: bool
	questions_per_second: float
	peak_memory_bytes: int


def choose_largest_suitable_batch(
	measurements: Sequence[BatchMeasurement],
	total_memory_bytes: int,
	minimum_peak_throughput_fraction: float = 0.95,
	memory_headroom_fraction: float = 0.10,
) -> int:
	"""Choose the largest near-peak batch that leaves requested CUDA headroom."""
	if total_memory_bytes <= 0:
		raise ValueError("total_memory_bytes must be positive")
	if not 0.0 < minimum_peak_throughput_fraction <= 1.0:
		raise ValueError("minimum_peak_throughput_fraction must be in (0, 1]")
	if not 0.0 <= memory_headroom_fraction < 1.0:
		raise ValueError("memory_headroom_fraction must be in [0, 1)")
	memory_limit = total_memory_bytes * (1.0 - memory_headroom_fraction)
	eligible = [
		measurement
		for measurement in measurements
		if (
			measurement.succeeded
			and measurement.batch_size > 0
			and measurement.questions_per_second > 0
			and measurement.peak_memory_bytes <= memory_limit
		)
	]
	if not eligible:
		raise RuntimeError("no successful batch leaves the requested CUDA headroom")
	peak_throughput = max(item.questions_per_second for item in eligible)
	near_peak = [
		item
		for item in eligible
		if item.questions_per_second
		>= peak_throughput * minimum_peak_throughput_fraction
	]
	return max(item.batch_size for item in near_peak)
