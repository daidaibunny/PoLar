#!/usr/bin/env python3
"""Run one pinned, full-path LLaMA generation on the local Apple MPS backend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.evaluation import FULL_PATH, GenerationSettings, score_math_generation
from reconstruction.executor import load_frozen_llama_executor


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"


def parse_arguments() -> argparse.Namespace:
	"""Parse the deliberately small local probe configuration."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--question", default="What is 1+1?")
	parser.add_argument("--ground-truth", default="2")
	return parser.parse_args()


def main() -> int:
	"""Load the pinned model, run one exact baseline path, and print timings."""
	arguments = parse_arguments()
	import torch

	load_started = time.perf_counter()
	executor = load_frozen_llama_executor(
		model_id=MODEL_ID,
		model_revision=MODEL_REVISION,
		tokenizer_revision=MODEL_REVISION,
		device="mps",
	)
	load_seconds = time.perf_counter() - load_started

	generation_started = time.perf_counter()
	batch = executor.generate(
		question=arguments.question,
		path=FULL_PATH,
		settings=GenerationSettings(),
	)
	generation_seconds = time.perf_counter() - generation_started
	generated_answer = batch.generated_answers[0]
	score = score_math_generation(
		question=arguments.question,
		ground_truth=arguments.ground_truth,
		generated_answer=generated_answer,
	)
	print(
		json.dumps(
			{
				"model_id": MODEL_ID,
				"model_revision": MODEL_REVISION,
				"device": "mps",
				"dtype": "bfloat16",
				"path": list(FULL_PATH),
				"max_new_tokens": 50,
				"load_seconds": load_seconds,
				"generation_seconds": generation_seconds,
				"mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
				"mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
				"generated_answer": generated_answer,
				"binary_reward": score.binary_reward,
			},
			ensure_ascii=False,
			indent=2,
			sort_keys=True,
		),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
