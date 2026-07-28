#!/usr/bin/env python3
"""Resume the fixed V2 label run with two independent workers per GPU."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, IO, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.label_generation import read_completed_question_ids
from scripts.run_mcts_labels_2gpu import (
	GPU_IDS,
	_merge_difficulty,
	_validate_idle_a800_gpus,
	_validate_public_data,
	build_label_command,
	build_validation_command,
)


DIFFICULTIES = tuple(range(1, 6))
WORKERS_PER_GPU = 2
EXPECTED_QUESTIONS_PER_SHARD = 50
LOSSLESS_OPTIMIZATIONS = (
	"batch_official_reward_evaluation_per_search_round",
	"two_independent_difficulty_workers_per_gpu",
)


@dataclass(frozen=True)
class ShardTask:
	"""One difficulty and fixed physical-GPU shard with isolated output files."""

	gpu_id: int
	difficulty: int
	trace_path: Path
	cache_path: Path
	log_path: Path
	command: Tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--model-revision", required=True)
	parser.add_argument("--hf-home", type=Path, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	parser.add_argument("--data-directory", type=Path, required=True)
	parser.add_argument("--expected-original-code-commit", required=True)
	parser.add_argument("--batch-size", type=int, default=50)
	parser.add_argument("--samples-per-difficulty", type=int, default=100)
	arguments = parser.parse_args(argv)
	if arguments.batch_size != 50:
		parser.error("the V2 optimized resume requires the original batch size 50")
	if arguments.samples_per_difficulty != 100:
		parser.error("the V2 optimized resume requires 100 questions per difficulty")
	if len(arguments.model_revision) != 40:
		parser.error("--model-revision must be a 40-character commit SHA")
	if len(arguments.expected_original_code_commit) != 40:
		parser.error("--expected-original-code-commit must be a 40-character SHA")
	return arguments


def build_shard_tasks(
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
	model_revision: str,
	batch_size: int,
	data_directory: Path,
	samples_per_difficulty: int,
) -> Dict[int, Tuple[ShardTask, ...]]:
	"""Build every unfinished task without changing its original CLI settings."""
	tasks: Dict[int, Tuple[ShardTask, ...]] = {}
	for gpu_id in GPU_IDS:
		gpu_tasks = []
		for difficulty in DIFFICULTIES:
			difficulty_root = output_root / f"diff-{difficulty}"
			trace_path = difficulty_root / f"trace-shard-{gpu_id}-of-2.jsonl"
			completed_count = len(read_completed_question_ids(trace_path))
			if completed_count > EXPECTED_QUESTIONS_PER_SHARD:
				raise RuntimeError(
					f"too many completed questions in {trace_path}: {completed_count}",
				)
			if completed_count == EXPECTED_QUESTIONS_PER_SHARD:
				continue
			gpu_tasks.append(
				ShardTask(
					gpu_id=gpu_id,
					difficulty=difficulty,
					trace_path=trace_path,
					cache_path=(
						difficulty_root / f"cache-shard-{gpu_id}-of-2.jsonl"
					),
					log_path=(
						output_root
						/ "logs"
						/ f"optimized-gpu-{gpu_id}-diff-{difficulty}.log"
					),
					command=build_label_command(
						python_executable=python_executable,
						repository_root=repository_root,
						output_root=output_root,
						model_revision=model_revision,
						batch_size=batch_size,
						difficulty=difficulty,
						shard_index=gpu_id,
						data_directory=data_directory,
						splits=("train", "validation", "test"),
						samples_per_difficulty=samples_per_difficulty,
						allow_test_oracle=True,
					),
				),
			)
		tasks[gpu_id] = tuple(gpu_tasks)
	return tasks


def validate_existing_run(
	output_root: Path,
	model_revision: str,
	batch_size: int,
	data_directory: Path,
	samples_per_difficulty: int,
	expected_original_code_commit: str,
) -> Dict[str, object]:
	"""Require an exact match with the already-started V2 result configuration."""
	manifest_path = output_root / "run_manifest.json"
	if not manifest_path.is_file():
		raise FileNotFoundError(f"missing existing V2 manifest: {manifest_path}")
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	expected_values = {
		"code_commit": expected_original_code_commit,
		"model_revision": model_revision,
		"tokenizer_revision": model_revision,
		"batch_size": batch_size,
		"seed": 42,
		"n_simulations": 200,
		"max_new_tokens": 50,
		"generation": "greedy",
		"length_penalty_lambda": 5.0,
		"random_action_probability": 0.1,
		"samples_per_difficulty": samples_per_difficulty,
		"splits": ["train", "validation", "test"],
		"test_oracle_diagnostic": True,
		"data_directory": str(data_directory),
	}
	for key, expected in expected_values.items():
		if manifest.get(key) != expected:
			raise RuntimeError(
				f"existing V2 manifest mismatch for {key}: "
				f"{manifest.get(key)!r} != {expected!r}",
			)
	expected_data_files = _validate_public_data(
		data_directory=data_directory,
		splits=("train", "validation", "test"),
		expected_samples_per_difficulty=samples_per_difficulty,
	)
	if manifest.get("data_files") != expected_data_files:
		raise RuntimeError("existing V2 manifest data file hashes do not match")
	return manifest


def record_lossless_resume(
	output_root: Path,
	manifest: Dict[str, object],
	execution_commit: str,
) -> None:
	"""Back up and atomically record the non-result-affecting execution change."""
	manifest_path = output_root / "run_manifest.json"
	original_commit = str(manifest["code_commit"])
	backup_path = manifest_path.with_name(
		f"run_manifest.json.pre-optimized-{original_commit[:7]}.bak",
	)
	if not backup_path.exists():
		shutil.copy2(manifest_path, backup_path)
	history = manifest.setdefault("lossless_resume_history", [])
	if not isinstance(history, list):
		raise RuntimeError("lossless_resume_history must be a list")
	entry = {
		"execution_commit": execution_commit,
		"original_code_commit": original_commit,
		"started_at_utc": datetime.now(timezone.utc).isoformat(),
		"workers_per_gpu": WORKERS_PER_GPU,
		"gpu_ids": list(GPU_IDS),
		"optimizations": list(LOSSLESS_OPTIMIZATIONS),
		"result_affecting_change": False,
	}
	if not any(item.get("execution_commit") == execution_commit for item in history):
		history.append(entry)
	temporary_path = manifest_path.with_name(
		f".{manifest_path.name}.optimized-{os.getpid()}.tmp",
	)
	with temporary_path.open("x", encoding="utf-8") as destination:
		json.dump(manifest, destination, indent=2, sort_keys=True)
		destination.write("\n")
		destination.flush()
		os.fsync(destination.fileno())
	os.replace(temporary_path, manifest_path)


def run_task(task: ShardTask, repository_root: Path, environment: Dict[str, str]) -> None:
	"""Run one unchanged shard command and require its complete 50-question trace."""
	task.log_path.parent.mkdir(parents=True, exist_ok=True)
	with task.log_path.open("a", encoding="utf-8") as log:
		log.write(json.dumps({
			"event": "optimized_shard_start",
			"gpu_id": task.gpu_id,
			"difficulty": task.difficulty,
			"command": task.command,
			"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		}) + "\n")
		log.flush()
		subprocess.run(
			task.command,
			cwd=repository_root,
			env={**environment, "CUDA_VISIBLE_DEVICES": str(task.gpu_id)},
			stdout=log,
			stderr=subprocess.STDOUT,
			check=True,
		)
	completed_count = len(read_completed_question_ids(task.trace_path))
	if completed_count != EXPECTED_QUESTIONS_PER_SHARD:
		raise RuntimeError(
			f"incomplete optimized task {task.trace_path}: {completed_count} questions",
		)
	print(json.dumps({
		"event": "optimized_shard_complete",
		"gpu_id": task.gpu_id,
		"difficulty": task.difficulty,
		"completed_questions": completed_count,
	}, sort_keys=True), flush=True)


def run_tasks(
	tasks_by_gpu: Dict[int, Tuple[ShardTask, ...]],
	repository_root: Path,
	environment: Dict[str, str],
) -> None:
	"""Keep two independent difficulty workers queued on each physical GPU."""
	executors = {
		gpu_id: concurrent.futures.ThreadPoolExecutor(
			max_workers=WORKERS_PER_GPU,
			thread_name_prefix=f"gpu-{gpu_id}",
		)
		for gpu_id in GPU_IDS
	}
	futures = []
	try:
		for gpu_id in GPU_IDS:
			futures.extend(
				executors[gpu_id].submit(
					run_task,
					task,
					repository_root,
					environment,
				)
				for task in tasks_by_gpu[gpu_id]
			)
		for future in concurrent.futures.as_completed(futures):
			future.result()
	finally:
		for executor in executors.values():
			executor.shutdown(wait=True, cancel_futures=True)


def acquire_lock(path: Path) -> IO[str]:
	"""Hold one non-blocking process lock for the optimized resume scheduler."""
	path.parent.mkdir(parents=True, exist_ok=True)
	stream = path.open("a+", encoding="utf-8")
	try:
		fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
	except BlockingIOError:
		stream.close()
		raise RuntimeError(f"another optimized resume owns {path}") from None
	return stream


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	repository_root = arguments.repository_root.resolve()
	output_root = arguments.output_root.resolve()
	data_directory = arguments.data_directory.resolve()
	hf_home = arguments.hf_home.resolve()
	if not hf_home.is_dir():
		raise FileNotFoundError(f"Hugging Face cache does not exist: {hf_home}")
	python_executable = repository_root / ".venv" / "bin" / "python"
	if not python_executable.is_file():
		raise FileNotFoundError(f"missing project Python: {python_executable}")
	lock = acquire_lock(output_root / ".optimized-resume.lock")
	try:
		manifest = validate_existing_run(
			output_root=output_root,
			model_revision=arguments.model_revision,
			batch_size=arguments.batch_size,
			data_directory=data_directory,
			samples_per_difficulty=arguments.samples_per_difficulty,
			expected_original_code_commit=arguments.expected_original_code_commit,
		)
		gpu_state = _validate_idle_a800_gpus(GPU_IDS)
		execution_commit = subprocess.check_output(
			["git", "rev-parse", "HEAD"],
			cwd=repository_root,
			text=True,
		).strip()
		record_lossless_resume(output_root, manifest, execution_commit)
		tasks_by_gpu = build_shard_tasks(
			python_executable=python_executable,
			repository_root=repository_root,
			output_root=output_root,
			model_revision=arguments.model_revision,
			batch_size=arguments.batch_size,
			data_directory=data_directory,
			samples_per_difficulty=arguments.samples_per_difficulty,
		)
		print(json.dumps({
			"event": "optimized_resume_start",
			"execution_commit": execution_commit,
			"gpu_state_before_launch": gpu_state,
			"workers_per_gpu": WORKERS_PER_GPU,
			"task_counts": {
				str(gpu_id): len(tasks_by_gpu[gpu_id]) for gpu_id in GPU_IDS
			},
		}, sort_keys=True), flush=True)
		environment = dict(os.environ)
		environment.update({
			"HF_HOME": str(hf_home),
			"HF_HUB_OFFLINE": "1",
			"TRANSFORMERS_OFFLINE": "1",
			"HF_DATASETS_OFFLINE": "1",
			"PYTHONUNBUFFERED": "1",
		})
		run_tasks(tasks_by_gpu, repository_root, environment)
		for difficulty in DIFFICULTIES:
			_merge_difficulty(
				python_executable=python_executable,
				repository_root=repository_root,
				output_root=output_root,
				difficulty=difficulty,
				resume=True,
				data_directory=data_directory,
				splits=("train", "validation", "test"),
				samples_per_difficulty=arguments.samples_per_difficulty,
				allow_test_oracle=True,
			)
		subprocess.run(
			build_validation_command(
				python_executable=python_executable,
				repository_root=repository_root,
				output_root=output_root,
			),
			cwd=repository_root,
			check=True,
		)
		print(json.dumps({
			"event": "optimized_resume_complete",
			"output_root": str(output_root),
		}, sort_keys=True), flush=True)
		return 0
	finally:
		lock.close()


if __name__ == "__main__":
	raise SystemExit(main())
