#!/usr/bin/env python3
"""Build the 7,500-query public PoLar reconstruction dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.redm_builder import (
	DatasetSource,
	SourceFile,
	build_redm_public,
)
from reconstruction.redm_sources import load_math_train_archive, validate_math_archive


QUERY_INFO_REPOSITORY = "hkust-nlp/dart-math-pool-math-query-info"
QUERY_INFO_REVISION = "e9415a3ed9a5b96f1738abc797dbe34c02dd790e"
DART_POOL_REPOSITORY = "hkust-nlp/dart-math-pool-math"
DART_POOL_REVISION = "2493eec006a329336a70255fcc93b8ff362a6aa5"
MATH_ARCHIVE_REPOSITORY = "https://gitee.com/hf-datasets/competition_math"
MATH_ARCHIVE_REVISION = "71b758ecc688b2822d07ffa7f8393299f1dc7cac"
MATH_ARCHIVE_SHA256 = "d9b88da85e6ffa3e1057ae675238d6e192574243bdc45ca7d00a1339fc4d0874"
EXPECTED_UNIQUE_QUESTIONS = 7_500


def parse_arguments() -> argparse.Namespace:
	"""Parse deterministic public reconstruction options."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output-directory",
		type=Path,
		default=REPOSITORY_ROOT / "data" / "redm-public",
	)
	parser.add_argument(
		"--math-archive",
		type=Path,
		default=REPOSITORY_ROOT / "data" / "sources" / "MATH.zip",
	)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def resolve_source(repository: str, revision: str) -> DatasetSource:
	"""Resolve immutable source data files and their LFS SHA-256 values."""
	try:
		from huggingface_hub import HfApi
	except ImportError as error:
		raise RuntimeError(
			"huggingface_hub is required to resolve source file hashes",
		) from error

	information = HfApi().dataset_info(
		repo_id=repository,
		revision=revision,
		files_metadata=True,
	)
	if information.sha != revision:
		raise RuntimeError(
			f"Dataset revision drift for {repository}: {information.sha} != {revision}",
		)
	files = []
	for sibling in information.siblings:
		if not sibling.rfilename.endswith((".parquet", ".json", ".jsonl")):
			continue
		lfs = sibling.lfs
		sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
		if not sha256:
			raise RuntimeError(
				f"Source data file has no verifiable SHA-256: {sibling.rfilename}",
			)
		files.append(
			SourceFile(
				path=sibling.rfilename,
				sha256=sha256,
				size_bytes=int(sibling.size),
			),
		)
	if not files:
		raise RuntimeError(f"No data files found for {repository}@{revision}")
	return DatasetSource(repository=repository, revision=revision, files=tuple(files))


def stream_rows(repository: str, revision: str) -> Iterable[dict[str, Any]]:
	"""Stream pinned DART-Math query metadata."""
	try:
		from datasets import load_dataset
	except ImportError as error:
		raise RuntimeError("datasets is required to stream DART-Math") from error
	return load_dataset(
		repository,
		split="train",
		revision=revision,
		streaming=True,
	)


def main() -> int:
	"""Build, hash, and report all five public difficulty files."""
	arguments = parse_arguments()
	query_source = resolve_source(QUERY_INFO_REPOSITORY, QUERY_INFO_REVISION)
	correction_source = resolve_source(DART_POOL_REPOSITORY, DART_POOL_REVISION)
	validate_math_archive(arguments.math_archive, MATH_ARCHIVE_SHA256)
	math_source = DatasetSource(
		repository=MATH_ARCHIVE_REPOSITORY,
		revision=MATH_ARCHIVE_REVISION,
		files=(
			SourceFile(
				path=arguments.math_archive.name,
				sha256=MATH_ARCHIVE_SHA256,
				size_bytes=arguments.math_archive.stat().st_size,
			),
		),
	)
	result = build_redm_public(
		query_info_rows=stream_rows(QUERY_INFO_REPOSITORY, QUERY_INFO_REVISION),
		question_rows=load_math_train_archive(arguments.math_archive),
		output_directory=arguments.output_directory,
		sources=(query_source, math_source, correction_source),
		seed=arguments.seed,
		expected_unique_questions=EXPECTED_UNIQUE_QUESTIONS,
	)
	print(
		json.dumps(
			{
				"manifest": str(result.manifest_path),
				"unique_questions": result.unique_questions,
				"difficulty_counts": result.level_counts,
			},
			sort_keys=True,
		),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
