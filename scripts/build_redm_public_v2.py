#!/usr/bin/env python3
"""Derive the five-by-100 ReDM-Public V2 dataset from fixed V1 files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.redm_builder import build_redm_public_v2


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	"""Parse fixed-size V2 derivation options."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--parent-directory",
		type=Path,
		default=Path("data/redm-public"),
	)
	parser.add_argument(
		"--output-directory",
		type=Path,
		default=Path("data/redm-public-v2"),
	)
	parser.add_argument("--questions-per-difficulty", type=int, default=100)
	parser.add_argument("--seed", type=int, default=42)
	arguments = parser.parse_args(argv)
	if arguments.questions_per_difficulty <= 0:
		parser.error("--questions-per-difficulty must be positive")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	"""Build V2 and print the exact output inventory."""
	arguments = parse_arguments(argv)
	result = build_redm_public_v2(
		parent_directory=arguments.parent_directory,
		output_directory=arguments.output_directory,
		questions_per_difficulty=arguments.questions_per_difficulty,
		seed=arguments.seed,
	)
	print(
		json.dumps(
			{
				"dataset_name": "ReDM-Public-V2-100",
				"manifest": str(result.manifest_path),
				"questions": result.unique_questions,
				"questions_by_difficulty": result.level_counts,
			},
			sort_keys=True,
		),
		flush=True,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
