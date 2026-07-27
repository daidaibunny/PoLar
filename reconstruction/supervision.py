"""Serialize executed MCTS results into official Predictor supervision files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

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
