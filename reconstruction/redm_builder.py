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
	question_rows: Iterable[Mapping[str, Any]],
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

	deduplicated = deduplicate_query_records(query_info_rows, question_rows)
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
			"source_question_rows": deduplicated.stats.pool_rows,
			"duplicate_source_question_rows": deduplicated.stats.duplicate_response_rows,
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


def build_redm_public_v2(
	parent_directory: Path,
	output_directory: Path,
	questions_per_difficulty: int = 100,
	seed: int = 42,
) -> BuildResult:
	"""Derive five fixed-size, domain-stratified bands from ReDM-Public V1."""
	if questions_per_difficulty <= 0:
		raise ValueError("questions_per_difficulty must be positive")
	parent_directory = Path(parent_directory)
	output_directory = Path(output_directory)
	output_paths = [output_directory / f"diff-{level}.json" for level in range(1, 6)]
	manifest_path = output_directory / "manifest.json"
	conflicting_outputs = [
		path for path in output_paths + [manifest_path] if path.exists()
	]
	if conflicting_outputs:
		raise FileExistsError(
			"Refusing to overwrite existing ReDM-Public V2 files: "
			+ ", ".join(str(path) for path in conflicting_outputs),
		)

	parent_manifest_path = parent_directory / "manifest.json"
	try:
		parent_manifest_bytes = parent_manifest_path.read_bytes()
		parent_manifest = json.loads(parent_manifest_bytes)
	except (FileNotFoundError, json.JSONDecodeError) as error:
		raise DataIntegrityError(
			f"Cannot read parent manifest {parent_manifest_path}: {error}",
		) from error
	if parent_manifest.get("dataset_name") != "ReDM-Public":
		raise DataIntegrityError(
			"V2 parent must be the fixed ReDM-Public dataset",
		)

	prepared_payloads: Dict[int, Dict[str, Any]] = {}
	level_manifest: Dict[str, Dict[str, Any]] = {}
	selected_records_by_level: Dict[int, Tuple[Dict[str, Any], ...]] = {}
	all_query_ids = set()
	for level in range(1, 6):
		level_metadata = parent_manifest.get("levels", {}).get(str(level), {})
		parent_file_name = level_metadata.get("output_file", f"diff-{level}.json")
		parent_path = parent_directory / str(parent_file_name)
		expected_sha256 = str(level_metadata.get("output_sha256", ""))
		actual_sha256 = sha256_file(parent_path)
		if actual_sha256 != expected_sha256:
			raise DataIntegrityError(
				f"Parent level hash mismatch for {parent_path}: "
				f"{actual_sha256} != {expected_sha256}",
			)
		try:
			parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
		except json.JSONDecodeError as error:
			raise DataIntegrityError(f"Invalid parent JSON: {parent_path}") from error
		if int(parent_payload.get("difficulty_level", level)) != level:
			raise DataIntegrityError(f"Parent difficulty mismatch in {parent_path}")

		available_records = []
		for split_name in ("train", "validation", "test"):
			stored_records = parent_payload.get(split_name)
			if not isinstance(stored_records, list):
				raise DataIntegrityError(
					f"Parent split {split_name!r} is not a list in {parent_path}",
				)
			for stored_record in stored_records:
				record = dict(stored_record)
				record["parent_split"] = split_name
				query_id = str(record.get("query_id", ""))
				if not query_id or query_id in all_query_ids:
					raise DataIntegrityError(
						f"Empty or duplicate parent query_id: {query_id!r}",
					)
				all_query_ids.add(query_id)
				available_records.append(record)
		if len(available_records) < questions_per_difficulty:
			raise DataIntegrityError(
				f"Difficulty {level} has only {len(available_records)} questions; "
				f"V2 requires {questions_per_difficulty}",
			)
		available_records.sort(key=lambda record: str(record["query_id"]))
		split = split_level_records(
			available_records,
			seed=seed,
			maximum_questions=questions_per_difficulty,
		)
		selected_records = tuple(split.train + split.validation + split.test)
		selected_records_by_level[level] = selected_records
		pass_rates = [float(record["dart_pass_rate"]) for record in selected_records]
		split_counts = {
			"train": len(split.train),
			"validation": len(split.validation),
			"test": len(split.test),
		}
		prepared_payloads[level] = {
			"difficulty_name": f"ReDM-Public-V2-{level}",
			"difficulty_level": level,
			"difficulty_definition": parent_payload.get("difficulty_definition"),
			"parent_dataset": "ReDM-Public",
			"parent_output_sha256": actual_sha256,
			"subset_seed": seed,
			"subset_strategy": (
				"domain-stratified fixed-size selection from the corresponding "
				"ReDM-Public V1 difficulty band"
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
		level_manifest[str(level)] = {
			"available_count": split.available_count,
			"selected_count": split.selected_count,
			"split_counts": split_counts,
			"pass_rate_range": prepared_payloads[level]["pass_rate_range"],
			"domain_counts": _domain_counts(selected_records),
			"output_file": f"diff-{level}.json",
		}

	output_directory.mkdir(parents=True, exist_ok=True)
	for level, output_path in zip(range(1, 6), output_paths):
		_write_json(output_path, prepared_payloads[level])
		level_manifest[str(level)]["output_sha256"] = sha256_file(output_path)

	selected_total = sum(len(records) for records in selected_records_by_level.values())
	manifest = {
		"schema_version": 2,
		"dataset_name": "ReDM-Public-V2-100",
		"reproduction_claim": "independent reconstruction",
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"seed": seed,
		"questions_per_difficulty": questions_per_difficulty,
		"selected_questions_total": selected_total,
		"difficulty_definition": parent_manifest.get("difficulty_definition"),
		"difficulty_order": parent_manifest.get("difficulty_order"),
		"selection_strategy": (
			"domain-stratified fixed-size selection within each fixed V1 difficulty band"
		),
		"split_ratios": {
			"train": 0.625,
			"validation": 0.125,
			"test": 0.25,
		},
		"intended_label_scope": ["train", "validation", "test"],
		"evaluation_warning": (
			"All V2 splits are intended for oracle MCTS labeling; V2 test records "
			"must not be used as an unsearched online evaluation set."
		),
		"parent_dataset": {
			"dataset_name": "ReDM-Public",
			"manifest_path": str(parent_manifest_path),
			"manifest_sha256": sha256_file(parent_manifest_path),
		},
		"sources": parent_manifest.get("sources", []),
		"statistics": {
			"selected_questions": selected_total,
			"by_public_difficulty": {
				str(level): len(records)
				for level, records in selected_records_by_level.items()
			},
			"by_domain": _domain_counts(
				record
				for records in selected_records_by_level.values()
				for record in records
			),
		},
		"levels": level_manifest,
	}
	_write_json(manifest_path, manifest)
	return BuildResult(
		output_directory=output_directory,
		manifest_path=manifest_path,
		unique_questions=selected_total,
		level_counts={
			level: len(records)
			for level, records in selected_records_by_level.items()
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
