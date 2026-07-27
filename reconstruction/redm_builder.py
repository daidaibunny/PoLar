"""Build the five ReDM-Human difficulty files from canonical query records."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from reconstruction.data import (
	DataIntegrityError,
	assign_pass_rate_difficulty_bands,
	deduplicate_query_records,
	sha256_file,
	split_level_records,
)


@dataclass(frozen=True)
class SourceFile:
	"""Resolved metadata for one immutable source dataset file."""

	path: str
	sha256: str
	size_bytes: int


@dataclass(frozen=True)
class DatasetSource:
	"""An immutable Hugging Face dataset snapshot."""

	repository: str
	revision: str
	files: Tuple[SourceFile, ...]


@dataclass(frozen=True)
class BuildResult:
	"""Paths and counts produced by a ReDM-Human build."""

	output_directory: Path
	manifest_path: Path
	unique_questions: int
	level_counts: Dict[int, int]


def build_redm_human(
	query_info_rows: Iterable[Mapping[str, Any]],
	pool_rows: Iterable[Mapping[str, Any]],
	output_directory: Path,
	sources: Sequence[DatasetSource],
	seed: int = 42,
) -> BuildResult:
	"""Deduplicate, stratify, write five levels, and create an audit manifest."""
	output_directory = Path(output_directory)
	output_paths = [output_directory / f"diff-{level}.json" for level in range(1, 6)]
	manifest_path = output_directory / "manifest.json"
	_conflicting_outputs = [path for path in output_paths + [manifest_path] if path.exists()]
	if _conflicting_outputs:
		raise FileExistsError(
			"Refusing to overwrite existing ReDM-Human files: "
			+ ", ".join(str(path) for path in _conflicting_outputs),
		)

	deduplicated = deduplicate_query_records(query_info_rows, pool_rows)
	records_by_level: Dict[int, List[Dict[str, Any]]] = {
		level: [] for level in range(1, 6)
	}
	for record in deduplicated.records:
		level = _normalize_level(record.get("level"))
		canonical_record = dict(record)
		canonical_record["level"] = level
		records_by_level[level].append(canonical_record)

	output_directory.mkdir(parents=True, exist_ok=True)
	level_manifest: Dict[str, Dict[str, Any]] = {}
	for level, output_path in zip(range(1, 6), output_paths):
		split = split_level_records(records_by_level[level], seed=seed)
		payload = {
			"difficulty_name": f"ReDM-Human-{level}",
			"difficulty_level": level,
			"difficulty_definition": "original MATH human level label",
			"split_seed": seed,
			"split_strategy": "domain-stratified 62.5/12.5/25",
			"available_count": split.available_count,
			"selected_count": split.selected_count,
			"train": list(split.train),
			"validation": list(split.validation),
			"test": list(split.test),
		}
		_write_json(output_path, payload)
		level_manifest[str(level)] = {
			"available_count": split.available_count,
			"selected_count": split.selected_count,
			"split_counts": {
				"train": len(split.train),
				"validation": len(split.validation),
				"test": len(split.test),
			},
			"domain_counts": _domain_counts(records_by_level[level]),
			"output_file": output_path.name,
			"output_sha256": sha256_file(output_path),
		}

	manifest = {
		"schema_version": 1,
		"dataset_name": "ReDM-Human",
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"seed": seed,
		"maximum_questions_per_level": 2_000,
		"split_ratios": {
			"train": 0.625,
			"validation": 0.125,
			"test": 0.25,
		},
		"sources": [_source_payload(source) for source in sources],
		"statistics": {
			"unique_questions": len(deduplicated.records),
			"query_info_rows": deduplicated.stats.query_info_rows,
			"pool_response_rows": deduplicated.stats.pool_rows,
			"duplicate_response_rows": deduplicated.stats.duplicate_response_rows,
			"unknown_query_rows": deduplicated.stats.unknown_query_rows,
			"missing_query_ids": list(deduplicated.stats.missing_query_ids),
			"missing_answers": deduplicated.stats.missing_answers,
			"by_level": {
				str(level): len(records_by_level[level]) for level in range(1, 6)
			},
			"by_domain": _domain_counts(deduplicated.records),
			"by_level_domain": {
				str(level): _domain_counts(records_by_level[level])
				for level in range(1, 6)
			},
		},
		"levels": level_manifest,
	}
	_write_json(manifest_path, manifest)

	return BuildResult(
		output_directory=output_directory,
		manifest_path=manifest_path,
		unique_questions=len(deduplicated.records),
		level_counts={level: len(records) for level, records in records_by_level.items()},
	)


def build_redm_public(
	query_info_rows: Iterable[Mapping[str, Any]],
	pool_rows: Iterable[Mapping[str, Any]],
	output_directory: Path,
	sources: Sequence[DatasetSource],
	seed: int = 42,
	expected_unique_questions: int | None = None,
) -> BuildResult:
	"""Build five equal public difficulty bands from DART-Math pass rates."""
	output_directory = Path(output_directory)
	output_paths = [output_directory / f"diff-{level}.json" for level in range(1, 6)]
	manifest_path = output_directory / "manifest.json"
	_conflicting_outputs = [
		path for path in output_paths + [manifest_path] if path.exists()
	]
	if _conflicting_outputs:
		raise FileExistsError(
			"Refusing to overwrite existing ReDM-Public files: "
			+ ", ".join(str(path) for path in _conflicting_outputs),
		)

	deduplicated = deduplicate_query_records(query_info_rows, pool_rows)
	if deduplicated.stats.missing_query_ids:
		raise DataIntegrityError(
			"Response pool is missing query IDs required by query-info: "
			+ ", ".join(deduplicated.stats.missing_query_ids[:10]),
		)
	if deduplicated.stats.missing_answers:
		raise DataIntegrityError(
			f"Response pool has {deduplicated.stats.missing_answers} missing answers",
		)
	if (
		expected_unique_questions is not None
		and len(deduplicated.records) != expected_unique_questions
	):
		raise DataIntegrityError(
			"Unexpected public query count: "
			f"{len(deduplicated.records)} != {expected_unique_questions}",
		)

	records = []
	for stored_record in deduplicated.records:
		record = dict(stored_record)
		record["math_human_level"] = _normalize_level(record["level"])
		records.append(record)
	records_by_level = assign_pass_rate_difficulty_bands(records, band_count=5)

	output_directory.mkdir(parents=True, exist_ok=True)
	level_manifest: Dict[str, Dict[str, Any]] = {}
	for level, output_path in zip(range(1, 6), output_paths):
		level_records = records_by_level[level]
		split = split_level_records(
			level_records,
			seed=seed,
			maximum_questions=len(level_records),
		)
		split_counts = {
			"train": len(split.train),
			"validation": len(split.validation),
			"test": len(split.test),
		}
		pass_rates = [float(record["dart_pass_rate"]) for record in level_records]
		payload = {
			"difficulty_name": f"ReDM-Public-{level}",
			"difficulty_level": level,
			"difficulty_definition": (
				"public DART-Math query-info pass-rate quintile; "
				"DM-1 easiest and DM-5 hardest; independent reconstruction"
			),
			"split_seed": seed,
			"split_strategy": "domain-stratified 62.5/12.5/25",
			"available_count": split.available_count,
			"selected_count": split.selected_count,
			"split_counts": split_counts,
			"pass_rate_range": {
				"minimum": min(pass_rates),
				"maximum": max(pass_rates),
			},
			"train": list(split.train),
			"validation": list(split.validation),
			"test": list(split.test),
		}
		_write_json(output_path, payload)
		level_manifest[str(level)] = {
			"available_count": split.available_count,
			"selected_count": split.selected_count,
			"split_counts": split_counts,
			"pass_rate_range": payload["pass_rate_range"],
			"domain_counts": _domain_counts(level_records),
			"output_file": output_path.name,
			"output_sha256": sha256_file(output_path),
		}

	manifest = {
		"schema_version": 1,
		"dataset_name": "ReDM-Public",
		"reproduction_claim": "independent reconstruction",
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"seed": seed,
		"difficulty_definition": (
			"five equal query-count bands sorted by descending DART-Math pass_rate; "
			"query_id breaks ties"
		),
		"difficulty_order": "DM-1 easiest to DM-5 hardest",
		"split_ratios": {
			"train": 0.625,
			"validation": 0.125,
			"test": 0.25,
		},
		"sources": [_source_payload(source) for source in sources],
		"statistics": {
			"unique_questions": len(deduplicated.records),
			"query_info_rows": deduplicated.stats.query_info_rows,
			"pool_response_rows": deduplicated.stats.pool_rows,
			"duplicate_response_rows": deduplicated.stats.duplicate_response_rows,
			"unknown_query_rows": deduplicated.stats.unknown_query_rows,
			"missing_query_ids": list(deduplicated.stats.missing_query_ids),
			"missing_answers": deduplicated.stats.missing_answers,
			"by_public_difficulty": {
				str(level): len(level_records)
				for level, level_records in records_by_level.items()
			},
			"by_math_human_level": dict(
				sorted(
					Counter(record["math_human_level"] for record in records).items(),
				),
			),
			"by_domain": _domain_counts(records),
		},
		"levels": level_manifest,
	}
	_write_json(manifest_path, manifest)

	return BuildResult(
		output_directory=output_directory,
		manifest_path=manifest_path,
		unique_questions=len(deduplicated.records),
		level_counts={
			level: len(level_records)
			for level, level_records in records_by_level.items()
		},
	)


def _normalize_level(value: Any) -> int:
	if isinstance(value, bool):
		raise DataIntegrityError(f"Invalid MATH level: {value!r}")
	if isinstance(value, int):
		level = value
	else:
		text = str(value).strip().lower().replace("level", "").strip()
		try:
			level = int(text)
		except ValueError as error:
			raise DataIntegrityError(f"Invalid MATH level: {value!r}") from error
	if level not in range(1, 6):
		raise DataIntegrityError(f"MATH level must be between 1 and 5: {value!r}")
	return level


def _domain_counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
	counts = Counter(str(record.get("domain", "")) for record in records)
	return dict(sorted(counts.items()))


def _source_payload(source: DatasetSource) -> Dict[str, Any]:
	return {
		"repository": source.repository,
		"revision": source.revision,
		"files": [asdict(source_file) for source_file in source.files],
	}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
	with path.open("x", encoding="utf-8") as destination:
		json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
		destination.write("\n")
