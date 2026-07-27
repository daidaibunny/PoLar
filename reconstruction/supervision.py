"""Serialize executed MCTS results into official Predictor supervision files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from reconstruction.mcts import SearchResult, is_predictor_compatible_path


def search_result_record(
	question_id: str,
	question: str,
	ground_truth: str,
	result: SearchResult,
	additional_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
	"""Create one auditable search record without inventing unexecuted paths."""
	metadata = dict(result.search_metadata)
	metadata.update(additional_metadata)
	return {
		"question_id": question_id,
		"question": question,
		"gt_ans": ground_truth,
		"initial_score": result.initial_score,
		"final_valid_transitions": [list(path) for path in result.valid_programs],
		"final_invalid_transitions": [list(path) for path in result.invalid_programs],
		"search_metadata": metadata,
	}


def predictor_supervision_record(
	record: Mapping[str, Any],
	original_depth: int,
) -> Dict[str, Any]:
	"""Keep only valid paths accepted by the official Predictor representation."""
	filtered_paths = []
	seen = set()
	for stored_path in record.get("final_valid_transitions", ()):
		path = tuple(int(layer) for layer in stored_path)
		if path in seen or not is_predictor_compatible_path(path, original_depth):
			continue
		seen.add(path)
		filtered_paths.append(list(path))
	result = dict(record)
	result["final_valid_transitions"] = filtered_paths
	result.pop("final_invalid_transitions", None)
	return result


def read_and_merge_trace_shards(
	paths: Sequence[Path],
	expected_question_ids: Sequence[str],
	original_depth: int,
) -> Tuple[Dict[str, Any], ...]:
	"""Merge append-only trace shards and require exact question-level coverage."""
	if not paths:
		raise ValueError("at least one trace shard is required")
	expected = tuple(str(question_id) for question_id in expected_question_ids)
	if any(not question_id for question_id in expected):
		raise ValueError("expected_question_ids cannot contain empty values")
	if len(expected) != len(set(expected)):
		raise ValueError("expected_question_ids must be unique")

	records_by_question_id: Dict[str, Dict[str, Any]] = {}
	for path in paths:
		path = Path(path)
		with path.open("r", encoding="utf-8") as source:
			for line_number, line in enumerate(source, start=1):
				if not line.strip():
					continue
				try:
					payload = json.loads(line)
					if not isinstance(payload, dict):
						raise TypeError("trace record must be a JSON object")
					question_id = str(payload["question_id"])
				except (json.JSONDecodeError, KeyError, TypeError) as error:
					raise ValueError(
						f"invalid trace record at {path}:{line_number}: {error}",
					) from error
				if not question_id or question_id in records_by_question_id:
					raise ValueError(
						f"empty or duplicate question_id at {path}:{line_number}",
					)
				records_by_question_id[question_id] = payload

	found = set(records_by_question_id)
	expected_set = set(expected)
	missing = sorted(expected_set - found)
	unexpected = sorted(found - expected_set)
	if missing or unexpected:
		raise ValueError(
			f"trace coverage mismatch: missing={missing}, unexpected={unexpected}",
		)
	return tuple(
		predictor_supervision_record(
			records_by_question_id[question_id],
			original_depth=original_depth,
		)
		for question_id in expected
	)


def validate_with_official_parser(
	records: Iterable[Mapping[str, Any]],
	original_depth: int,
) -> None:
	"""Reject any supervision path the official ``polar.data`` parser rejects."""
	try:
		from polar.data import parse_path_to_seg_and_ops
	except ImportError as error:
		raise RuntimeError("Official Predictor parser dependencies are unavailable") from error

	for record in records:
		question_id = str(record.get("question_id", ""))
		for stored_path in record.get("final_valid_transitions", ()):
			path = [int(layer) for layer in stored_path]
			parsed = parse_path_to_seg_and_ops(
				path,
				original_depth=original_depth,
				max_pack=4,
				allow_repeat=True,
			)
			if parsed is None:
				raise ValueError(
					f"Official parser rejected question_id={question_id!r}, path={path!r}",
				)


def write_merged_mcts_samples(
	path: Path,
	records: Sequence[Mapping[str, Any]],
	original_depth: int,
) -> None:
	"""Validate and write the official ``merged_mcts_samples.json`` container."""
	path = Path(path)
	if path.exists():
		raise FileExistsError(f"Refusing to overwrite existing supervision: {path}")
	validate_with_official_parser(records, original_depth)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("x", encoding="utf-8") as destination:
		json.dump(
			{"samples": list(records)},
			destination,
			ensure_ascii=False,
			indent=2,
			sort_keys=True,
		)
		destination.write("\n")
