"""Verified source readers for the public DART-Math reconstruction."""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

from reconstruction.data import DataIntegrityError, sha256_file


MATH_ANSWER_CORRECTIONS = {
	"MATH/train/algebra/24014.json": "2",
	"MATH/train/algebra/25040.json": "9",
	"MATH/train/number_theory/7115.json": "0",
	"MATH/train/number_theory/7117.json": "0",
}


def validate_math_archive(archive_path: Path, expected_sha256: str) -> None:
	"""Require the exact official MATH archive bytes before reading records."""
	archive_path = Path(archive_path)
	if not archive_path.is_file():
		raise FileNotFoundError(f"MATH archive does not exist: {archive_path}")
	actual_sha256 = sha256_file(archive_path)
	if actual_sha256 != expected_sha256:
		raise DataIntegrityError(
			f"MATH archive SHA-256 mismatch: {actual_sha256} != {expected_sha256}",
		)


def load_math_train_archive(archive_path: Path) -> Tuple[Dict[str, Any], ...]:
	"""Read canonical MATH train questions while preserving source query paths."""
	archive_path = Path(archive_path)
	if zipfile.is_zipfile(archive_path):
		return _load_math_train_zip(archive_path)
	if tarfile.is_tarfile(archive_path):
		return _load_math_train_tar(archive_path)
	raise DataIntegrityError(f"Unsupported MATH archive format: {archive_path}")


def _load_math_train_zip(archive_path: Path) -> Tuple[Dict[str, Any], ...]:
	"""Read public MATH records from the hash-pinned ZIP mirror."""
	records = []
	seen_query_ids = set()
	with zipfile.ZipFile(archive_path) as archive:
		for member in sorted(archive.infolist(), key=lambda item: item.filename):
			if member.is_dir():
				continue
			query_id = _canonical_train_query_id(member.filename)
			if query_id is None:
				continue
			if query_id in seen_query_ids:
				raise DataIntegrityError(f"Duplicate MATH archive path: {query_id}")
			seen_query_ids.add(query_id)
			with archive.open(member) as source:
				payload = json.load(source)
			records.append(_math_record(query_id, payload))
	return tuple(records)


def _load_math_train_tar(archive_path: Path) -> Tuple[Dict[str, Any], ...]:
	"""Read public MATH records from the original TAR layout."""
	records = []
	seen_query_ids = set()
	with tarfile.open(archive_path, "r:*") as archive:
		for member in sorted(archive.getmembers(), key=lambda item: item.name):
			if not member.isfile():
				continue
			query_id = _canonical_train_query_id(member.name)
			if query_id is None:
				continue
			if query_id in seen_query_ids:
				raise DataIntegrityError(f"Duplicate MATH archive path: {query_id}")
			seen_query_ids.add(query_id)
			source = archive.extractfile(member)
			if source is None:
				raise DataIntegrityError(f"Unable to read MATH archive path: {query_id}")
			try:
				payload = json.load(source)
			finally:
				source.close()
			records.append(_math_record(query_id, payload))
	return tuple(records)


def _canonical_train_query_id(member_name: str) -> str | None:
	query_path = PurePosixPath(member_name.removeprefix("./"))
	parts = query_path.parts
	if len(parts) != 4 or parts[:2] != ("MATH", "train"):
		return None
	if query_path.suffix != ".json":
		return None
	return query_path.as_posix()


def _math_record(query_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
	question = payload.get("problem")
	if not isinstance(question, str) or not question.strip():
		raise DataIntegrityError(f"Missing MATH problem text: {query_id}")
	answer = _extract_answer_from_math_solution(str(payload.get("solution", "")))
	answer_source = "MATH boxed solution"
	verified_correction = MATH_ANSWER_CORRECTIONS.get(query_id)
	if verified_correction is not None:
		if answer is not None and answer.strip() and answer.strip() != verified_correction:
			raise DataIntegrityError(
				"MATH answer disagrees with verified DART-Math correction for "
				f"{query_id}: {answer!r} != {verified_correction!r}",
			)
		answer = verified_correction
		answer_source = "DART-Math response pool"
	if answer is None or not answer.strip():
		raise DataIntegrityError(f"Missing boxed MATH answer: {query_id}")
	return {
		"query_id": query_id,
		"query": question,
		"gt_ans": answer,
		"level": _extract_math_level(payload),
		"domain": str(payload.get("type", "")),
		"answer_source": answer_source,
	}


def _extract_answer_from_math_solution(solution: str) -> str | None:
	"""Match the official DART-Math last-boxed-answer extraction."""
	start = solution.rfind("\\boxed")
	if start < 0:
		start = solution.rfind("\\fbox")
		if start < 0:
			return None
	open_braces = 0
	end = None
	for index in range(start, len(solution)):
		if solution[index] == "{":
			open_braces += 1
		elif solution[index] == "}":
			open_braces -= 1
			if open_braces == 0:
				end = index
				break
	if end is None:
		return None
	boxed = solution[start : end + 1]
	prefix = "\\boxed{"
	if not boxed.startswith(prefix) or not boxed.endswith("}"):
		return None
	return boxed[len(prefix) : -1]


def _extract_math_level(payload: Dict[str, Any]) -> int:
	"""Match the two unknown-level corrections in official DART-Math."""
	level = str(payload.get("level", "")).split(" ")[-1]
	if level == "?":
		level = "2" if str(payload.get("problem", "")).startswith("We") else "1"
	try:
		return int(level)
	except ValueError as error:
		raise DataIntegrityError(f"Invalid MATH level: {level!r}") from error
