"""ReDM-Human data validation, deduplication, splitting, and hashing."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


class DataIntegrityError(ValueError):
	"""Raised when source rows disagree about a question's canonical content."""


@dataclass(frozen=True)
class DeduplicationStats:
	"""Counters collected while joining query metadata to response rows."""

	query_info_rows: int
	unique_query_ids: int
	pool_rows: int
	duplicate_response_rows: int
	unknown_query_rows: int
	missing_query_ids: Tuple[str, ...]
	missing_answers: int


@dataclass(frozen=True)
class DeduplicationResult:
	"""Canonical question records and their integrity statistics."""

	records: Tuple[Dict[str, Any], ...]
	stats: DeduplicationStats


@dataclass(frozen=True)
class LevelSplit:
	"""A deterministic, domain-stratified split for one difficulty level."""

	train: Tuple[Dict[str, Any], ...]
	validation: Tuple[Dict[str, Any], ...]
	test: Tuple[Dict[str, Any], ...]
	available_count: int
	selected_count: int


def deduplicate_query_records(
	query_info_rows: Iterable[Mapping[str, Any]],
	pool_rows: Iterable[Mapping[str, Any]],
) -> DeduplicationResult:
	"""Join query metadata to one canonical question and answer per query ID.

	The query-info table defines the eligible question IDs and supplies level,
	domain, and DART-Math pass rate. The response pool is scanned in full because
	each question may have several generated responses. Repeated response rows are
	not treated as additional questions.
	"""
	metadata_by_id: Dict[str, Dict[str, Any]] = {}
	query_info_count = 0
	for row in query_info_rows:
		query_info_count += 1
		query_id = _required_string(row, "query_id")
		metadata = {
			"level": row.get("level"),
			"domain": row.get("domain"),
			"dart_pass_rate": row.get("pass_rate"),
		}
		previous = metadata_by_id.get(query_id)
		if previous is not None and previous != metadata:
			raise DataIntegrityError(
				f"Conflicting query-info rows for query_id={query_id!r}",
			)
		metadata_by_id[query_id] = metadata

	canonical_pool_rows: Dict[str, Tuple[Any, Any]] = {}
	pool_count = 0
	duplicate_count = 0
	unknown_count = 0
	for row in pool_rows:
		pool_count += 1
		query_id = _required_string(row, "query_id")
		if query_id not in metadata_by_id:
			unknown_count += 1
			continue
		question = row.get("query")
		answer = row.get("gt_ans")
		candidate = (question, answer)
		previous = canonical_pool_rows.get(query_id)
		if previous is None:
			canonical_pool_rows[query_id] = candidate
			continue
		duplicate_count += 1
		if previous != candidate:
			raise DataIntegrityError(
				"Conflicting response-pool question or answer for "
				f"query_id={query_id!r}: {previous!r} != {candidate!r}",
			)

	missing_query_ids = tuple(sorted(set(metadata_by_id) - set(canonical_pool_rows)))
	records: List[Dict[str, Any]] = []
	missing_answers = 0
	for query_id in sorted(canonical_pool_rows):
		question, answer = canonical_pool_rows[query_id]
		if answer is None or not str(answer).strip():
			missing_answers += 1
		metadata = metadata_by_id[query_id]
		records.append(
			{
				"query_id": query_id,
				"question": question,
				"gt_ans": answer,
				"level": metadata["level"],
				"domain": metadata["domain"],
				"dart_pass_rate": metadata["dart_pass_rate"],
			},
		)

	return DeduplicationResult(
		records=tuple(records),
		stats=DeduplicationStats(
			query_info_rows=query_info_count,
			unique_query_ids=len(metadata_by_id),
			pool_rows=pool_count,
			duplicate_response_rows=duplicate_count,
			unknown_query_rows=unknown_count,
			missing_query_ids=missing_query_ids,
			missing_answers=missing_answers,
		),
	)


def split_level_records(
	records: Sequence[Mapping[str, Any]],
	seed: int = 42,
	maximum_questions: int = 2_000,
) -> LevelSplit:
	"""Make a capped 62.5/12.5/25 split while preserving domain proportions."""
	if maximum_questions <= 0:
		raise ValueError("maximum_questions must be positive")

	canonical_records = [dict(record) for record in records]
	_query_ids_are_unique(canonical_records)
	groups = _group_by_domain(canonical_records)
	selected_count = min(len(canonical_records), maximum_questions)
	selected = _stratified_select(groups, selected_count, seed)
	selected_groups = _group_by_domain(selected)

	targets = _largest_remainder(selected_count, (0.625, 0.125, 0.25))
	allocations = _allocate_domain_splits(selected_groups, targets)
	splits: List[List[Dict[str, Any]]] = [[], [], []]
	for domain in sorted(selected_groups):
		domain_records = list(selected_groups[domain])
		_stable_shuffle(domain_records, seed, f"split:{domain}")
		offset = 0
		for split_index, count in enumerate(allocations[domain]):
			splits[split_index].extend(domain_records[offset : offset + count])
			offset += count

	for split_index, split_records in enumerate(splits):
		_stable_shuffle(split_records, seed, f"output:{split_index}")

	return LevelSplit(
		train=tuple(splits[0]),
		validation=tuple(splits[1]),
		test=tuple(splits[2]),
		available_count=len(canonical_records),
		selected_count=selected_count,
	)


def sha256_file(path: Path) -> str:
	"""Return the SHA-256 digest of a file's exact bytes."""
	digest = hashlib.sha256()
	with Path(path).open("rb") as source:
		for chunk in iter(lambda: source.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _required_string(row: Mapping[str, Any], key: str) -> str:
	value = row.get(key)
	if value is None or not str(value).strip():
		raise DataIntegrityError(f"Missing required {key!r} in row: {row!r}")
	return str(value)


def _query_ids_are_unique(records: Sequence[Mapping[str, Any]]) -> None:
	seen = set()
	for record in records:
		query_id = _required_string(record, "query_id")
		if query_id in seen:
			raise DataIntegrityError(f"Duplicate query_id in split input: {query_id!r}")
		seen.add(query_id)


def _group_by_domain(
	records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
	groups: Dict[str, List[Dict[str, Any]]] = {}
	for record in records:
		domain = str(record.get("domain", ""))
		groups.setdefault(domain, []).append(record)
	return groups


def _stratified_select(
	groups: Mapping[str, Sequence[Dict[str, Any]]],
	count: int,
	seed: int,
) -> List[Dict[str, Any]]:
	total = sum(len(group) for group in groups.values())
	if count == total:
		return [dict(record) for group in groups.values() for record in group]
	quotas = _largest_remainder(
		count,
		tuple(len(groups[domain]) / total for domain in sorted(groups)),
	)
	selected: List[Dict[str, Any]] = []
	for domain, quota in zip(sorted(groups), quotas):
		domain_records = [dict(record) for record in groups[domain]]
		_stable_shuffle(domain_records, seed, f"select:{domain}")
		selected.extend(domain_records[:quota])
	return selected


def _allocate_domain_splits(
	groups: Mapping[str, Sequence[Dict[str, Any]]],
	targets: Tuple[int, ...],
) -> Dict[str, Tuple[int, ...]]:
	ratios = (0.625, 0.125, 0.25)
	domains = sorted(groups)
	allocations = {
		domain: [math.floor(len(groups[domain]) * ratio) for ratio in ratios]
		for domain in domains
	}
	domain_remaining = {
		domain: len(groups[domain]) - sum(allocations[domain]) for domain in domains
	}
	split_remaining = [
		targets[index] - sum(allocations[domain][index] for domain in domains)
		for index in range(3)
	]

	candidates = []
	for domain in domains:
		for split_index, ratio in enumerate(ratios):
			exact = len(groups[domain]) * ratio
			fraction = exact - math.floor(exact)
			candidates.append((-fraction, domain, split_index))
	candidates.sort()

	while sum(split_remaining) > 0:
		progress = False
		for _, domain, split_index in candidates:
			if domain_remaining[domain] <= 0 or split_remaining[split_index] <= 0:
				continue
			allocations[domain][split_index] += 1
			domain_remaining[domain] -= 1
			split_remaining[split_index] -= 1
			progress = True
		if not progress:
			raise DataIntegrityError("Unable to satisfy exact stratified split sizes")

	return {domain: tuple(values) for domain, values in allocations.items()}


def _largest_remainder(total: int, weights: Tuple[float, ...]) -> Tuple[int, ...]:
	if not weights:
		return ()
	weight_sum = sum(weights)
	if total == 0:
		return tuple(0 for _ in weights)
	if weight_sum <= 0:
		raise ValueError("weights must have a positive sum")
	exact = [total * weight / weight_sum for weight in weights]
	result = [math.floor(value) for value in exact]
	remaining = total - sum(result)
	order = sorted(range(len(weights)), key=lambda index: (-(exact[index] % 1), index))
	for index in order[:remaining]:
		result[index] += 1
	return tuple(result)


def _stable_shuffle(records: List[Dict[str, Any]], seed: int, namespace: str) -> None:
	seed_material = f"{seed}:{namespace}".encode("utf-8")
	derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
	random.Random(derived_seed).shuffle(records)

