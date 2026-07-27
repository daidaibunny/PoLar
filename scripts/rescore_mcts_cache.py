#!/usr/bin/env python3
"""Re-score cached MCTS generations with the official PoLar batch evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.cache import JsonlEvaluationCache
from reconstruction.evaluation import MathScoreInput, score_math_generations
from reconstruction.mcts import EvaluationResult


ScoreGenerations = Callable[
	[Sequence[MathScoreInput]],
	Sequence[EvaluationResult],
]


def rescore_cache(
	cache_path: Path,
	data_file: Path,
	batch_size: int,
	backup_suffix: str,
	score_generations: ScoreGenerations = score_math_generations,
) -> Dict[str, object]:
	"""Atomically replace rewards after retaining a byte-identical backup."""
	cache_path = Path(cache_path)
	data_file = Path(data_file)
	if batch_size <= 0:
		raise ValueError("batch_size must be positive")
	if not backup_suffix or "/" in backup_suffix:
		raise ValueError("backup_suffix must be a non-empty filename component")
	if not cache_path.is_file():
		raise FileNotFoundError(cache_path)

	validated_cache = JsonlEvaluationCache(cache_path)
	validated_cache.close()
	payloads = tuple(
		json.loads(line)
		for line in cache_path.read_text(encoding="utf-8").splitlines()
		if line.strip()
	)
	questions = _load_questions(data_file)
	updated_payloads = [dict(payload) for payload in payloads]
	changed_rewards = 0

	for offset in range(0, len(payloads), batch_size):
		batch = payloads[offset : offset + batch_size]
		score_inputs = tuple(
			_score_input(payload, questions)
			for payload in batch
		)
		evaluations = tuple(score_generations(score_inputs))
		if len(evaluations) != len(batch):
			raise RuntimeError("official scorer returned an unexpected result count")
		for relative_index, (payload, evaluation) in enumerate(
			zip(batch, evaluations, strict=True),
		):
			if evaluation.generated_answer != payload["generated_answer"]:
				raise RuntimeError("official scorer changed generated-answer ordering")
			absolute_index = offset + relative_index
			new_reward = evaluation.binary_reward
			changed_rewards += int(new_reward != int(payload["binary_reward"]))
			updated_payloads[absolute_index]["binary_reward"] = new_reward

	backup_path = cache_path.with_name(
		f"{cache_path.name}.pre-official-{backup_suffix}.bak",
	)
	temporary_path = cache_path.with_name(f"{cache_path.name}.official.tmp")
	if backup_path.exists():
		raise FileExistsError(backup_path)
	if temporary_path.exists():
		raise FileExistsError(temporary_path)
	shutil.copy2(cache_path, backup_path)
	with temporary_path.open("x", encoding="utf-8") as destination:
		for payload in updated_payloads:
			destination.write(json.dumps(payload, sort_keys=True) + "\n")
		destination.flush()
		os.fsync(destination.fileno())

	new_cache = JsonlEvaluationCache(temporary_path)
	new_cache.close()
	os.replace(temporary_path, cache_path)
	return {
		"cache_path": str(cache_path),
		"backup_path": str(backup_path),
		"entries": len(payloads),
		"changed_rewards": changed_rewards,
		"old_sha256": hashlib.sha256(backup_path.read_bytes()).hexdigest(),
		"new_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
	}


def _load_questions(data_file: Path) -> Dict[str, Tuple[str, str]]:
	payload = json.loads(data_file.read_text(encoding="utf-8"))
	questions: Dict[str, Tuple[str, str]] = {}
	for split in ("train", "validation"):
		for record in payload[split]:
			question_id = str(record["query_id"])
			if question_id in questions:
				raise ValueError(f"duplicate query_id in data file: {question_id}")
			questions[question_id] = (
				str(record["question"]),
				str(record["gt_ans"]),
			)
	return questions


def _score_input(
	payload: Dict[str, object],
	questions: Dict[str, Tuple[str, str]],
) -> MathScoreInput:
	question_id = str(payload["identity"]["question_id"])
	if question_id not in questions:
		raise ValueError(f"cache question_id is absent from data file: {question_id}")
	question, ground_truth = questions[question_id]
	return question, ground_truth, str(payload["generated_answer"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--cache-jsonl", type=Path, required=True)
	parser.add_argument("--data-file", type=Path, required=True)
	parser.add_argument("--batch-size", type=int, default=192)
	parser.add_argument("--backup-suffix", required=True)
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	summary = rescore_cache(
		cache_path=arguments.cache_jsonl,
		data_file=arguments.data_file,
		batch_size=arguments.batch_size,
		backup_suffix=arguments.backup_suffix,
	)
	print(json.dumps(summary, sort_keys=True), flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
