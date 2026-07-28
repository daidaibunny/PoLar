#!/usr/bin/env python3
"""Generate append-only MCTS traces with batched frozen LLaMA execution."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.cache import JsonlEvaluationCache
from reconstruction.evaluation import GenerationSettings, MODEL_ID, ORIGINAL_DEPTH
from reconstruction.executor import load_frozen_llama_executor
from reconstruction.label_generation import (
	append_records_jsonl,
	generate_mcts_records,
	load_search_samples,
	read_completed_question_ids,
)
from reconstruction.mcts import MCTSConfig, SearchMode


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", type=Path, required=True)
	parser.add_argument(
		"--splits",
		nargs="+",
		choices=("train", "validation", "test"),
		default=["train", "validation"],
	)
	parser.add_argument("--output-jsonl", type=Path, required=True)
	parser.add_argument("--cache-jsonl", type=Path, required=True)
	parser.add_argument("--model-id", default=MODEL_ID)
	parser.add_argument("--model-revision")
	parser.add_argument("--tokenizer-revision")
	parser.add_argument(
		"--mode",
		choices=tuple(mode.value for mode in SearchMode),
		default=SearchMode.PREDICTOR_COMPATIBLE.value,
	)
	parser.add_argument("--n-simulations", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--search-width", type=int, default=16)
	parser.add_argument("--max-samples", type=int, default=20)
	parser.add_argument("--num-shards", type=int, default=1)
	parser.add_argument("--shard-index", type=int, default=0)
	parser.add_argument("--max-new-tokens", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--ucb-c", type=float, default=2**0.5)
	parser.add_argument("--length-penalty-lambda", type=float, default=5.0)
	parser.add_argument("--random-action-probability", type=float, default=0.1)
	parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
	parser.add_argument("--allow-test-oracle", action="store_true")
	arguments = parser.parse_args(argv)
	if "test" in arguments.splits and not arguments.allow_test_oracle:
		parser.error("test search requires --allow-test-oracle and is diagnostic only")
	if arguments.n_simulations < 0:
		parser.error("--n-simulations must be non-negative")
	if arguments.batch_size <= 0:
		parser.error("--batch-size must be positive")
	if arguments.search_width < arguments.batch_size:
		parser.error("--search-width must be at least --batch-size")
	if arguments.max_samples <= 0:
		parser.error("--max-samples must be positive")
	if arguments.num_shards <= 0:
		parser.error("--num-shards must be positive")
	if arguments.shard_index < 0 or arguments.shard_index >= arguments.num_shards:
		parser.error("--shard-index must be within [0, --num-shards)")
	return arguments


def resolve_model_revision(model_id: str, requested_revision: Optional[str]) -> str:
	"""Resolve a model reference to an immutable Hugging Face commit SHA."""
	if requested_revision and re.fullmatch(r"[0-9a-f]{40}", requested_revision):
		return requested_revision
	try:
		from huggingface_hub import HfApi

		model_info = HfApi().model_info(model_id, revision=requested_revision)
	except Exception as error:
		raise RuntimeError(
			f"Cannot access {model_id!r}. Configure gated-model access with "
			"`hf auth login` in this environment, then retry.",
		) from error
	if not model_info.sha:
		raise RuntimeError(f"Hugging Face did not return a commit SHA for {model_id!r}")
	return str(model_info.sha)


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	samples = load_search_samples(
		path=arguments.data_file,
		splits=arguments.splits,
		max_samples=arguments.max_samples,
		num_shards=arguments.num_shards,
		shard_index=arguments.shard_index,
	)
	completed = read_completed_question_ids(arguments.output_jsonl)
	remaining = tuple(sample for sample in samples if sample.question_id not in completed)
	if not remaining:
		print(json.dumps({"status": "complete", "completed_questions": len(completed)}))
		return 0

	model_revision = resolve_model_revision(arguments.model_id, arguments.model_revision)
	tokenizer_revision = arguments.tokenizer_revision or model_revision
	executor = load_frozen_llama_executor(
		model_id=arguments.model_id,
		model_revision=model_revision,
		tokenizer_revision=tokenizer_revision,
		device=arguments.device,
	)
	settings = GenerationSettings(max_new_tokens=arguments.max_new_tokens)
	search_config = MCTSConfig(
		n_simulations=arguments.n_simulations,
		exploration_constant=arguments.ucb_c,
		length_penalty_lambda=arguments.length_penalty_lambda,
		random_action_probability=arguments.random_action_probability,
		seed=arguments.seed,
	)
	cache = JsonlEvaluationCache(arguments.cache_jsonl)
	start_time = time.monotonic()
	total_cache_hits = 0
	total_cache_misses = 0
	total_model_batches = 0
	total_reward_evaluator_batches = 0

	for offset in range(0, len(remaining), arguments.search_width):
		chunk = remaining[offset : offset + arguments.search_width]
		result = generate_mcts_records(
			samples=chunk,
			executor=executor,
			cache=cache,
			search_config=search_config,
			search_mode=SearchMode(arguments.mode),
			model_id=arguments.model_id,
			model_revision=model_revision,
			tokenizer_revision=tokenizer_revision,
			batch_size=arguments.batch_size,
			original_depth=ORIGINAL_DEPTH,
			settings=settings,
			run_metadata={
				"num_shards": arguments.num_shards,
				"shard_index": arguments.shard_index,
				"test_oracle_diagnostic": "test" in arguments.splits,
			},
		)
		append_records_jsonl(arguments.output_jsonl, result.records)
		total_cache_hits += result.cache_hits
		total_cache_misses += result.cache_misses
		total_model_batches += result.model_batches
		total_reward_evaluator_batches += result.reward_evaluator_batches
		print(
			json.dumps(
				{
					"status": "progress",
					"completed_this_run": min(offset + len(chunk), len(remaining)),
					"remaining_this_run": max(0, len(remaining) - offset - len(chunk)),
					"cache_hits": total_cache_hits,
					"cache_misses": total_cache_misses,
					"model_batches": total_model_batches,
					"reward_evaluator_batches": total_reward_evaluator_batches,
				},
				sort_keys=True,
			),
			flush=True,
		)
	cache.close()

	elapsed_seconds = time.monotonic() - start_time
	peak_memory_bytes = _peak_gpu_memory_bytes(arguments.device)
	summary = {
		"status": "complete",
		"questions": len(remaining),
		"cache_hits": total_cache_hits,
		"executed_paths": total_cache_misses,
		"model_batches": total_model_batches,
		"reward_evaluator_batches": total_reward_evaluator_batches,
		"elapsed_seconds": elapsed_seconds,
		"seconds_per_executed_path": (
			elapsed_seconds / total_cache_misses if total_cache_misses else None
		),
		"peak_gpu_memory_bytes": peak_memory_bytes,
		"model_revision": model_revision,
		"tokenizer_revision": tokenizer_revision,
	}
	print(json.dumps(summary, sort_keys=True), flush=True)
	return 0


def _peak_gpu_memory_bytes(device: str) -> Optional[int]:
	if device != "cuda":
		return None
	try:
		import torch

		return int(torch.cuda.max_memory_allocated())
	except Exception:
		return None


if __name__ == "__main__":
	raise SystemExit(main())
