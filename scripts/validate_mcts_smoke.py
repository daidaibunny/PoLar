#!/usr/bin/env python3
"""Validate that a Predictor-compatible MCTS smoke reaches useful programs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from polar.config import OP_REPEAT, OP_SKIP
from polar.data import parse_path_to_seg_and_ops


PROGRAM_KINDS = ("standard", "skip_only", "repeat_only", "skip_repeat")
NONSTANDARD_PROGRAM_KINDS = PROGRAM_KINDS[1:]


def classify_predictor_path(path: Sequence[int], original_depth: int) -> str:
	"""Classify one path using the official Predictor parser's canonical labels."""
	parsed = parse_path_to_seg_and_ops(
		[int(layer) for layer in path],
		original_depth=original_depth,
		max_pack=4,
		allow_repeat=True,
	)
	if parsed is None:
		raise ValueError(f"official Predictor parser rejected path={list(path)!r}")
	_, operation_labels = parsed
	has_skip = OP_SKIP in operation_labels
	has_repeat = OP_REPEAT in operation_labels
	if has_skip and has_repeat:
		return "skip_repeat"
	if has_skip:
		return "skip_only"
	if has_repeat:
		return "repeat_only"
	return "standard"


def summarize_trace_records(
	records: Iterable[Mapping[str, Any]],
	original_depth: int,
) -> Dict[str, Any]:
	"""Summarize raw traces without treating an evaluated wrong path as supervision."""
	if original_depth <= 0:
		raise ValueError("original_depth must be positive")
	evaluated_counts: Counter[str] = Counter()
	valid_counts: Counter[str] = Counter()
	questions_with_valid_kind: Counter[str] = Counter()
	question_ids = set()
	parser_failures = []
	duplicate_path_count = 0
	incompatible_mode_records = 0
	records_depth_gt_one = 0
	records_selection_positive = 0
	maximum_depth = 0
	total_selection_count = 0
	record_count = 0

	for record_index, record in enumerate(records):
		record_count += 1
		question_id = str(record.get("question_id", ""))
		if not question_id or question_id in question_ids:
			raise ValueError(
				f"empty or duplicate question_id at record index {record_index}: "
				f"{question_id!r}",
			)
		question_ids.add(question_id)
		metadata = record.get("search_metadata", {})
		if not isinstance(metadata, Mapping):
			raise ValueError(f"search_metadata must be an object for {question_id!r}")
		if metadata.get("mode") != "predictor_compatible":
			incompatible_mode_records += 1
		depth = int(metadata.get("maximum_tree_depth_reached", 0))
		selection_count = int(metadata.get("tree_policy_selection_count", 0))
		maximum_depth = max(maximum_depth, depth)
		total_selection_count += selection_count
		if depth > 1:
			records_depth_gt_one += 1
		if selection_count > 0:
			records_selection_positive += 1

		seen_paths = set()
		valid_kinds_for_question = set()
		for field, is_valid in (
			("final_valid_transitions", True),
			("final_invalid_transitions", False),
		):
			stored_paths = record.get(field, ())
			if not isinstance(stored_paths, (list, tuple)):
				raise ValueError(f"{field} must be a list for {question_id!r}")
			for stored_path in stored_paths:
				path = tuple(int(layer) for layer in stored_path)
				if path in seen_paths:
					duplicate_path_count += 1
					continue
				seen_paths.add(path)
				try:
					kind = classify_predictor_path(path, original_depth)
				except ValueError as error:
					parser_failures.append(
						{"question_id": question_id, "path": list(path), "error": str(error)},
					)
					continue
				evaluated_counts[kind] += 1
				if is_valid:
					valid_counts[kind] += 1
					valid_kinds_for_question.add(kind)
		for kind in valid_kinds_for_question:
			questions_with_valid_kind[kind] += 1

	return {
		"questions": record_count,
		"evaluated_path_counts": _complete_counts(evaluated_counts),
		"valid_path_counts": _complete_counts(valid_counts),
		"questions_with_valid_kind": _complete_counts(questions_with_valid_kind),
		"parser_failure_count": len(parser_failures),
		"parser_failure_examples": parser_failures[:5],
		"duplicate_path_count": duplicate_path_count,
		"incompatible_mode_records": incompatible_mode_records,
		"records_depth_gt_one": records_depth_gt_one,
		"records_selection_positive": records_selection_positive,
		"maximum_tree_depth_reached": maximum_depth,
		"total_tree_policy_selection_count": total_selection_count,
	}


def smoke_failures(
	summary: Mapping[str, Any],
	required_valid_kinds: Sequence[str] = NONSTANDARD_PROGRAM_KINDS,
) -> Tuple[str, ...]:
	"""Return every failed hard gate for a real-model MCTS smoke."""
	failures = []
	questions = int(summary.get("questions", 0))
	if questions <= 0:
		failures.append("trace contains no questions")
	if int(summary.get("parser_failure_count", 0)):
		failures.append("at least one evaluated path failed the official parser")
	if int(summary.get("duplicate_path_count", 0)):
		failures.append("at least one question evaluated a duplicate path")
	if int(summary.get("incompatible_mode_records", 0)):
		failures.append("at least one record was not predictor_compatible")
	if int(summary.get("records_depth_gt_one", 0)) != questions:
		failures.append("not every question reached tree depth greater than one")
	if int(summary.get("records_selection_positive", 0)) != questions:
		failures.append("not every question selected an explored child")

	evaluated_counts = summary.get("evaluated_path_counts", {})
	valid_counts = summary.get("valid_path_counts", {})
	for kind in NONSTANDARD_PROGRAM_KINDS:
		if int(evaluated_counts.get(kind, 0)) <= 0:
			failures.append(f"no evaluated {kind} path was found")
	for kind in required_valid_kinds:
		if kind not in NONSTANDARD_PROGRAM_KINDS:
			raise ValueError(f"unsupported required valid kind: {kind!r}")
		if int(valid_counts.get(kind, 0)) <= 0:
			failures.append(f"no valid {kind} path was found")
	return tuple(failures)


def load_trace_records(path: Path) -> Tuple[Mapping[str, Any], ...]:
	"""Load an append-only JSONL trace with line-specific errors."""
	records = []
	with Path(path).open("r", encoding="utf-8") as source:
		for line_number, line in enumerate(source, start=1):
			if not line.strip():
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as error:
				raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
			if not isinstance(record, dict):
				raise ValueError(f"trace record must be an object at {path}:{line_number}")
			records.append(record)
	return tuple(records)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--trace-jsonl", type=Path, nargs="+", required=True)
	parser.add_argument("--original-depth", type=int, default=28)
	parser.add_argument("--summary-json", type=Path)
	parser.add_argument(
		"--require-valid-kind",
		action="append",
		choices=NONSTANDARD_PROGRAM_KINDS,
		dest="required_valid_kinds",
	)
	arguments = parser.parse_args(argv)
	if arguments.original_depth <= 0:
		parser.error("--original-depth must be positive")
	if arguments.required_valid_kinds is None:
		arguments.required_valid_kinds = list(NONSTANDARD_PROGRAM_KINDS)
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	records = tuple(
		record
		for trace_path in arguments.trace_jsonl
		for record in load_trace_records(trace_path)
	)
	summary = summarize_trace_records(records, arguments.original_depth)
	failures = smoke_failures(summary, arguments.required_valid_kinds)
	result = dict(summary)
	result["status"] = "passed" if not failures else "failed"
	result["failures"] = list(failures)
	serialized = json.dumps(result, indent=2, sort_keys=True)
	print(serialized, flush=True)
	if arguments.summary_json is not None:
		if arguments.summary_json.exists():
			raise FileExistsError(
				f"refusing to overwrite smoke summary: {arguments.summary_json}",
			)
		arguments.summary_json.parent.mkdir(parents=True, exist_ok=True)
		arguments.summary_json.write_text(serialized + "\n", encoding="utf-8")
	return 0 if not failures else 1


def _complete_counts(counts: Mapping[str, int]) -> Dict[str, int]:
	return {kind: int(counts.get(kind, 0)) for kind in PROGRAM_KINDS}


if __name__ == "__main__":
	raise SystemExit(main())
