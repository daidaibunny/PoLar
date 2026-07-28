#!/usr/bin/env python3
"""Merge MCTS trace shards into official Predictor supervision JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.evaluation import ORIGINAL_DEPTH
from reconstruction.label_generation import load_search_samples
from reconstruction.supervision import (
	read_and_merge_trace_shards,
	write_merged_mcts_samples,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", type=Path, required=True)
	parser.add_argument(
		"--splits",
		nargs="+",
		choices=("train", "validation", "test"),
		default=["train", "validation"],
	)
	parser.add_argument("--allow-test-oracle", action="store_true")
	parser.add_argument("--trace-jsonl", type=Path, nargs="+", required=True)
	parser.add_argument("--output-json", type=Path, required=True)
	parser.add_argument("--max-samples", type=int, default=20)
	parser.add_argument("--original-depth", type=int, default=ORIGINAL_DEPTH)
	arguments = parser.parse_args(argv)
	if arguments.max_samples <= 0:
		parser.error("--max-samples must be positive")
	if arguments.original_depth <= 0:
		parser.error("--original-depth must be positive")
	if "test" in arguments.splits and not arguments.allow_test_oracle:
		parser.error("test labels require --allow-test-oracle")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	samples = load_search_samples(
		path=arguments.data_file,
		splits=arguments.splits,
		max_samples=arguments.max_samples,
		num_shards=1,
		shard_index=0,
	)
	records = read_and_merge_trace_shards(
		paths=arguments.trace_jsonl,
		expected_question_ids=tuple(sample.question_id for sample in samples),
		original_depth=arguments.original_depth,
	)
	write_merged_mcts_samples(
		path=arguments.output_json,
		records=records,
		original_depth=arguments.original_depth,
	)
	valid_path_counts = [len(record["final_valid_transitions"]) for record in records]
	print(
		json.dumps(
			{
				"status": "complete",
				"output_json": str(arguments.output_json),
				"questions": len(records),
				"valid_paths": sum(valid_path_counts),
				"questions_without_valid_paths": sum(
					count == 0 for count in valid_path_counts
				),
			},
			sort_keys=True,
		),
		flush=True,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
