#!/usr/bin/env python3
"""Measure stable greedy-generation batch sizes on one assigned CUDA GPU."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.batching import BatchMeasurement, choose_largest_suitable_batch
from reconstruction.evaluation import FULL_PATH, MODEL_ID, GenerationSettings
from reconstruction.executor import load_frozen_llama_executor
from reconstruction.label_generation import load_search_samples
from scripts.generate_mcts_labels import resolve_model_revision


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", type=Path, required=True)
	parser.add_argument("--output-json", type=Path, required=True)
	parser.add_argument("--model-id", default=MODEL_ID)
	parser.add_argument("--model-revision")
	parser.add_argument("--tokenizer-revision")
	parser.add_argument(
		"--batch-sizes",
		type=int,
		nargs="+",
		default=[8, 16, 32, 64, 128],
	)
	parser.add_argument("--max-samples", type=int, default=128)
	parser.add_argument("--max-new-tokens", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--memory-headroom-fraction", type=float, default=0.10)
	parser.add_argument("--minimum-peak-throughput-fraction", type=float, default=0.95)
	arguments = parser.parse_args(argv)
	if arguments.output_json.exists():
		parser.error(f"refusing to overwrite {arguments.output_json}")
	if arguments.max_samples <= 0:
		parser.error("--max-samples must be positive")
	if any(batch_size <= 0 for batch_size in arguments.batch_sizes):
		parser.error("--batch-sizes values must be positive")
	if max(arguments.batch_sizes) > arguments.max_samples:
		parser.error("--max-samples must be at least the largest batch size")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	try:
		import numpy
		import torch
	except ImportError as error:
		raise RuntimeError("CUDA benchmark dependencies are unavailable") from error
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for batch-size benchmarking")
	random.seed(arguments.seed)
	numpy.random.seed(arguments.seed)
	torch.manual_seed(arguments.seed)
	torch.cuda.manual_seed_all(arguments.seed)

	samples = load_search_samples(
		path=arguments.data_file,
		splits=("train",),
		max_samples=arguments.max_samples,
		num_shards=1,
		shard_index=0,
	)
	questions = tuple(sample.question for sample in samples)
	model_revision = resolve_model_revision(arguments.model_id, arguments.model_revision)
	tokenizer_revision = arguments.tokenizer_revision or model_revision
	executor = load_frozen_llama_executor(
		model_id=arguments.model_id,
		model_revision=model_revision,
		tokenizer_revision=tokenizer_revision,
		device="cuda",
	)
	settings = GenerationSettings(max_new_tokens=arguments.max_new_tokens)
	executor.generate_batch(questions[:1], path=FULL_PATH, settings=settings)
	torch.cuda.synchronize()
	measurements = []
	measurement_details = []
	for batch_size in arguments.batch_sizes:
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()
		start_time = time.monotonic()
		error_message = None
		try:
			for offset in range(0, len(questions), batch_size):
				executor.generate_batch(
					questions[offset : offset + batch_size],
					path=FULL_PATH,
					settings=settings,
				)
			torch.cuda.synchronize()
			succeeded = True
		except RuntimeError as error:
			if "out of memory" not in str(error).lower():
				raise
			torch.cuda.synchronize()
			succeeded = False
			error_message = "CUDA out of memory"
		elapsed_seconds = time.monotonic() - start_time
		questions_per_second = len(questions) / elapsed_seconds if succeeded else 0.0
		peak_allocated = int(torch.cuda.max_memory_allocated())
		peak_reserved = int(torch.cuda.max_memory_reserved())
		measurement = BatchMeasurement(
			batch_size=batch_size,
			succeeded=succeeded,
			questions_per_second=questions_per_second,
			peak_memory_bytes=peak_allocated,
		)
		measurements.append(measurement)
		measurement_details.append(
			{
				**asdict(measurement),
				"peak_reserved_memory_bytes": peak_reserved,
				"elapsed_seconds": elapsed_seconds,
				"error": error_message,
			},
		)
		print(json.dumps(measurement_details[-1], sort_keys=True), flush=True)

	device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
	selected_batch_size = choose_largest_suitable_batch(
		measurements=measurements,
		total_memory_bytes=int(device_properties.total_memory),
		minimum_peak_throughput_fraction=arguments.minimum_peak_throughput_fraction,
		memory_headroom_fraction=arguments.memory_headroom_fraction,
	)
	payload = {
		"selected_batch_size": selected_batch_size,
		"selection_rule": (
			"largest measured batch within "
			f"{arguments.minimum_peak_throughput_fraction:.1%} of eligible peak "
			"throughput and with at least "
			f"{arguments.memory_headroom_fraction:.1%} CUDA memory headroom"
		),
		"model_id": arguments.model_id,
		"model_revision": model_revision,
		"tokenizer_revision": tokenizer_revision,
		"gpu_name": device_properties.name,
		"total_memory_bytes": int(device_properties.total_memory),
		"seed": arguments.seed,
		"path": list(FULL_PATH),
		"generation": asdict(settings),
		"measurements": measurement_details,
	}
	arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
	with arguments.output_json.open("x", encoding="utf-8") as destination:
		json.dump(payload, destination, indent=2, sort_keys=True)
		destination.write("\n")
	print(json.dumps(payload, sort_keys=True), flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
