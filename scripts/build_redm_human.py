#!/usr/bin/env python3
"""Build ReDM-Human from pinned DART-Math Hugging Face revisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.redm_builder import DatasetSource, SourceFile, build_redm_human


QUERY_INFO_REPOSITORY = "hkust-nlp/dart-math-pool-math-query-info"
QUERY_INFO_REVISION = "e9415a3ed9a5b96f1738abc797dbe34c02dd790e"
POOL_REPOSITORY = "hkust-nlp/dart-math-pool-math"
POOL_REVISION = "2493eec006a329336a70255fcc93b8ff362a6aa5"


def parse_arguments() -> argparse.Namespace:
	"""Parse the deterministic dataset build options."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output-directory",
		type=Path,
		default=REPOSITORY_ROOT / "data" / "redm-human",
	)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def resolve_source(repository: str, revision: str) -> DatasetSource:
	"""Resolve LFS SHA-256 values for every immutable dataset data file."""
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
	"""Stream a pinned dataset split without materializing its response pool."""
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
	"""Resolve source metadata and build all five ReDM-Human files."""
	arguments = parse_arguments()
	query_source = resolve_source(QUERY_INFO_REPOSITORY, QUERY_INFO_REVISION)
	pool_source = resolve_source(POOL_REPOSITORY, POOL_REVISION)
	result = build_redm_human(
		query_info_rows=stream_rows(QUERY_INFO_REPOSITORY, QUERY_INFO_REVISION),
		pool_rows=stream_rows(POOL_REPOSITORY, POOL_REVISION),
		output_directory=arguments.output_directory,
		sources=(query_source, pool_source),
		seed=arguments.seed,
	)
	print(
		json.dumps(
			{
				"manifest": str(result.manifest_path),
				"unique_questions": result.unique_questions,
				"level_counts": result.level_counts,
			},
			sort_keys=True,
		),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
