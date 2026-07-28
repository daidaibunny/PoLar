#!/usr/bin/env python3
"""Build official-format Predictor data with held-out, unsearched test records."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.supervision import write_merged_mcts_samples


MODEL_RELATIVE_PATH = Path("meta-llama") / "Llama-3.2-3B-Instruct"
PUBLIC_SPLIT_COUNTS = {"train": 938, "validation": 187, "test": 375}


def prepare_difficulty(
	supervision_path: Path,
	data_path: Path,
	output_path: Path,
	original_depth: int,
	expected_split_counts: Mapping[str, int],
) -> Dict[str, object]:
	"""Combine searched supervision with held-out test records without oracle labels."""
	supervision_path = Path(supervision_path)
	data_path = Path(data_path)
	output_path = Path(output_path)
	supervision_bytes = supervision_path.read_bytes()
	data_bytes = data_path.read_bytes()
	supervision = json.loads(supervision_bytes)
	data = json.loads(data_bytes)
	supervised_records = supervision["samples"]

	source_by_id = {}
	ordered_ids = {"train": [], "validation": [], "test": []}
	for split in ordered_ids:
		rows = data[split]
		if len(rows) != int(expected_split_counts[split]):
			raise ValueError(
				f"unexpected {split} count in {data_path}: {len(rows)}",
			)
		for row in rows:
			question_id = str(row["query_id"])
			if not question_id or question_id in source_by_id:
				raise ValueError(f"duplicate or empty query_id in {data_path}: {question_id!r}")
			source_by_id[question_id] = (split, row)
			ordered_ids[split].append(question_id)

	supervised_by_id = {}
	for record in supervised_records:
		question_id = str(record["question_id"])
		if not question_id or question_id in supervised_by_id:
			raise ValueError(
				f"duplicate or empty supervised question_id: {question_id!r}",
			)
		if question_id not in source_by_id:
			raise ValueError(f"supervised question is absent from source data: {question_id}")
		split, source = source_by_id[question_id]
		if split == "test":
			raise ValueError(f"test oracle leakage in supervision: {question_id}")
		if record.get("search_metadata", {}).get("data_split") != split:
			raise ValueError(f"recorded split mismatch for {question_id}")
		if str(record["question"]) != str(source["question"]):
			raise ValueError(f"question mismatch for {question_id}")
		if str(record["gt_ans"]) != str(source["gt_ans"]):
			raise ValueError(f"ground-truth mismatch for {question_id}")
		supervised_by_id[question_id] = dict(record)

	expected_supervised_ids = set(ordered_ids["train"] + ordered_ids["validation"])
	if set(supervised_by_id) != expected_supervised_ids:
		missing = sorted(expected_supervised_ids - set(supervised_by_id))
		unexpected = sorted(set(supervised_by_id) - expected_supervised_ids)
		raise ValueError(
			f"supervision coverage mismatch: missing={missing}, unexpected={unexpected}",
		)

	output_records = [
		supervised_by_id[question_id]
		for split in ("train", "validation")
		for question_id in ordered_ids[split]
	]
	for question_id in ordered_ids["test"]:
		_source_split, source = source_by_id[question_id]
		output_records.append(
			{
				"question_id": question_id,
				"question": str(source["question"]),
				"gt_ans": str(source["gt_ans"]),
				"initial_score": None,
				"final_valid_transitions": [],
				"final_invalid_transitions": [],
				"search_metadata": {
					"data_split": "test",
					"mcts_searched": False,
					"oracle_labels_present": False,
				},
			},
		)

	write_merged_mcts_samples(
		path=output_path,
		records=output_records,
		original_depth=original_depth,
	)
	output_bytes = output_path.read_bytes()
	return {
		"supervision_path": str(supervision_path),
		"supervision_sha256": hashlib.sha256(supervision_bytes).hexdigest(),
		"source_data_path": str(data_path),
		"source_data_sha256": hashlib.sha256(data_bytes).hexdigest(),
		"output_path": str(output_path),
		"output_sha256": hashlib.sha256(output_bytes).hexdigest(),
		"split_counts": dict(expected_split_counts),
		"valid_paths": sum(
			len(record["final_valid_transitions"])
			for record in supervised_records
		),
		"test_oracle_labels": 0,
	}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--supervision-root", type=Path, required=True)
	parser.add_argument("--redm-data-root", type=Path, required=True)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	parser.add_argument("--original-depth", type=int, default=28)
	arguments = parser.parse_args(argv)
	if arguments.original_depth <= 0:
		parser.error("--original-depth must be positive")
	if arguments.output_root.exists():
		parser.error("--output-root already exists; use a new path")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	repository_root = arguments.repository_root.resolve()
	supervision_model_root = arguments.supervision_root.resolve() / MODEL_RELATIVE_PATH
	output_root = arguments.output_root.resolve()
	output_model_root = output_root / MODEL_RELATIVE_PATH
	summaries = {}
	for difficulty in range(1, 6):
		summaries[str(difficulty)] = prepare_difficulty(
			supervision_path=(
				supervision_model_root
				/ f"dart-math-diff-{difficulty}"
				/ "merged_mcts_samples.json"
			),
			data_path=arguments.redm_data_root.resolve() / f"diff-{difficulty}.json",
			output_path=(
				output_model_root
				/ f"dart-math-diff-{difficulty}"
				/ "merged_mcts_samples.json"
			),
			original_depth=arguments.original_depth,
			expected_split_counts=PUBLIC_SPLIT_COUNTS,
		)
	commit = subprocess.check_output(
		["git", "rev-parse", "HEAD"],
		cwd=repository_root,
		text=True,
	).strip()
	manifest = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"code_commit": commit,
		"model_id": "meta-llama/Llama-3.2-3B-Instruct",
		"original_depth": arguments.original_depth,
		"test_policy": "held-out records have no MCTS paths or oracle labels",
		"difficulties": summaries,
	}
	manifest_path = output_root / "predictor_data_manifest.json"
	with manifest_path.open("x", encoding="utf-8") as destination:
		json.dump(manifest, destination, indent=2, sort_keys=True)
		destination.write("\n")
	print(json.dumps({"status": "complete", "manifest": str(manifest_path)}))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
