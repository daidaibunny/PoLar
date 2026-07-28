#!/usr/bin/env python3
"""Run public Predictor-compatible MCTS labels on both or one assigned CUDA shard."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EXPECTED_SAMPLES_PER_DIFFICULTY = 1_125
GPU_IDS = (0, 1)
OFFICIAL_REWARD_EVALUATOR = {
	"implementation": "dart_math.eval.EvaluatorMathBatch",
	"strict_extract": True,
	"use_orig_eq_for_olympiadbench": True,
	"timeout_seconds": 60,
	"processes_per_gpu": 4,
	"query": "",
	"dataset": "math",
	"source": "official PoLar polar/eval.py::_batch_compare_answers",
}


def build_label_command(
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
	model_revision: str,
	batch_size: int,
	difficulty: int,
	shard_index: int,
	data_directory: Optional[Path] = None,
	splits: Sequence[str] = ("train", "validation"),
	samples_per_difficulty: int = EXPECTED_SAMPLES_PER_DIFFICULTY,
	allow_test_oracle: bool = False,
) -> Tuple[str, ...]:
	"""Build one fixed-parameter command for a difficulty and physical GPU shard."""
	if "test" in splits and not allow_test_oracle:
		raise ValueError("test labels require allow_test_oracle=True")
	data_root = data_directory or repository_root / "data" / "redm-public"
	difficulty_root = output_root / f"diff-{difficulty}"
	command = (
		str(python_executable),
		str(repository_root / "scripts" / "generate_mcts_labels.py"),
		"--data-file",
		str(data_root / f"diff-{difficulty}.json"),
		"--splits",
		*splits,
		"--output-jsonl",
		str(difficulty_root / f"trace-shard-{shard_index}-of-2.jsonl"),
		"--cache-jsonl",
		str(difficulty_root / f"cache-shard-{shard_index}-of-2.jsonl"),
		"--model-id",
		MODEL_ID,
		"--model-revision",
		model_revision,
		"--tokenizer-revision",
		model_revision,
		"--mode",
		"predictor_compatible",
		"--n-simulations",
		"200",
		"--batch-size",
		str(batch_size),
		"--search-width",
		str(batch_size),
		"--max-samples",
		str(samples_per_difficulty),
		"--num-shards",
		"2",
		"--shard-index",
		str(shard_index),
		"--max-new-tokens",
		"50",
		"--seed",
		"42",
		"--ucb-c",
		str(2**0.5),
		"--length-penalty-lambda",
		"5.0",
		"--random-action-probability",
		"0.1",
		"--device",
		"cuda",
	)
	if allow_test_oracle:
		command += ("--allow-test-oracle",)
	return command


def build_validation_command(
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
) -> Tuple[str, ...]:
	"""Build the cross-difficulty raw-trace validation command."""
	trace_paths = tuple(
		str(
			output_root
			/ f"diff-{difficulty}"
			/ f"trace-shard-{shard_index}-of-2.jsonl",
		)
		for difficulty in range(1, 6)
		for shard_index in GPU_IDS
	)
	return (
		str(python_executable),
		str(repository_root / "scripts" / "validate_mcts_smoke.py"),
		"--trace-jsonl",
		*trace_paths,
		"--original-depth",
		"28",
		"--summary-json",
		str(output_root / "label_validation_summary.json"),
	)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--model-revision", required=True)
	parser.add_argument("--batch-size", type=int, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	parser.add_argument(
		"--data-directory",
		type=Path,
		default=REPOSITORY_ROOT / "data" / "redm-public",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		choices=("train", "validation", "test"),
		default=["train", "validation"],
	)
	parser.add_argument(
		"--samples-per-difficulty",
		type=int,
		default=EXPECTED_SAMPLES_PER_DIFFICULTY,
	)
	parser.add_argument("--allow-test-oracle", action="store_true")
	parser.add_argument(
		"--only-shard",
		type=int,
		choices=GPU_IDS,
		help=(
			"Run only the selected fixed GPU/shard mapping. This permits staggered "
			"execution without touching the other physical GPU."
		),
	)
	parser.add_argument("--resume", action="store_true")
	arguments = parser.parse_args(argv)
	if not re.fullmatch(r"[0-9a-f]{40}", arguments.model_revision):
		parser.error("--model-revision must be a 40-character commit SHA")
	if arguments.batch_size <= 0:
		parser.error("--batch-size must be positive")
	if arguments.samples_per_difficulty <= 0:
		parser.error("--samples-per-difficulty must be positive")
	if "test" in arguments.splits and not arguments.allow_test_oracle:
		parser.error("test labels require --allow-test-oracle")
	if arguments.output_root.exists() and not arguments.resume:
		parser.error("--output-root already exists; use a new path or --resume")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	repository_root = arguments.repository_root.resolve()
	data_directory = arguments.data_directory.resolve()
	output_root = arguments.output_root.resolve()
	hf_home_value = os.environ.get("HF_HOME")
	if not hf_home_value:
		raise RuntimeError("HF_HOME must point to the shared Hugging Face cache")
	hf_home = Path(hf_home_value).resolve()
	if not hf_home.is_dir():
		raise RuntimeError(f"HF_HOME does not exist: {hf_home}")
	model_snapshot = (
		hf_home
		/ "hub"
		/ "models--meta-llama--Llama-3.2-3B-Instruct"
		/ "snapshots"
		/ arguments.model_revision
	)
	if not model_snapshot.is_dir():
		raise RuntimeError(f"pinned model snapshot is missing: {model_snapshot}")
	selected_gpu_ids = (
		GPU_IDS if arguments.only_shard is None else (arguments.only_shard,)
	)
	gpu_state = _validate_idle_a800_gpus(selected_gpu_ids)
	data_manifest = _validate_public_data(
		data_directory=data_directory,
		splits=arguments.splits,
		expected_samples_per_difficulty=arguments.samples_per_difficulty,
	)
	dataset_manifest_path = data_directory / "manifest.json"
	dataset_manifest_bytes = dataset_manifest_path.read_bytes()
	commit = subprocess.check_output(
		["git", "rev-parse", "HEAD"],
		cwd=repository_root,
		text=True,
	).strip()
	manifest = {
		"method_claim": "independent reconstruction",
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"code_commit": commit,
		"model_id": MODEL_ID,
		"model_revision": arguments.model_revision,
		"tokenizer_revision": arguments.model_revision,
		"parallelism": "two independent CUDA model replicas with question sharding",
		"physical_gpu_to_shard": {"0": 0, "1": 1},
		"staggered_shard_launch_supported": True,
		"gpu_state_before_launch": gpu_state,
		"batch_size": arguments.batch_size,
		"seed": 42,
		"n_simulations": 200,
		"max_new_tokens": 50,
		"generation": "greedy",
		"ucb_c": 2**0.5,
		"length_penalty_lambda": 5.0,
		"random_action_probability": 0.1,
		"tree_policy": (
			"expand one shuffled unexplored action when no child exists or with "
			"probability 0.1; otherwise select the highest-UCB explored child"
		),
		"max_block_length": 4,
		"max_repeat_count_predictor": 1,
		"reward_evaluator": OFFICIAL_REWARD_EVALUATOR,
		"lossless_optimizations": [
			"reuse_exact_batch_token_tensors",
			"reuse_prompt_hashes",
			"memoize_identical_scoring_inputs",
			"reuse_flush_per_record_cache_stream",
			"skip_network_lookup_for_pinned_revision",
		],
		"data_directory": str(data_directory),
		"splits": list(arguments.splits),
		"samples_per_difficulty": arguments.samples_per_difficulty,
		"test_oracle_diagnostic": "test" in arguments.splits,
		"dataset_manifest": {
			"path": str(dataset_manifest_path),
			"sha256": hashlib.sha256(dataset_manifest_bytes).hexdigest(),
		},
		"data_files": data_manifest,
	}
	_write_or_validate_manifest(output_root, manifest, arguments.resume)

	with concurrent.futures.ThreadPoolExecutor(
		max_workers=len(selected_gpu_ids),
	) as executor:
		futures = [
			executor.submit(
				_run_gpu_shard,
				gpu_id=gpu_id,
				python_executable=Path(sys.executable),
				repository_root=repository_root,
				output_root=output_root,
				model_revision=arguments.model_revision,
				batch_size=arguments.batch_size,
				data_directory=data_directory,
				splits=arguments.splits,
				samples_per_difficulty=arguments.samples_per_difficulty,
				allow_test_oracle=arguments.allow_test_oracle,
			)
			for gpu_id in selected_gpu_ids
		]
		for future in futures:
			future.result()

	if arguments.only_shard is not None:
		print(
			json.dumps(
				{
					"status": "shard_complete",
					"completed_shard": arguments.only_shard,
					"output_root": str(output_root),
				},
				sort_keys=True,
			),
			flush=True,
		)
		return 0

	for difficulty in range(1, 6):
		_merge_difficulty(
			python_executable=Path(sys.executable),
			repository_root=repository_root,
			output_root=output_root,
			difficulty=difficulty,
			resume=arguments.resume,
			data_directory=data_directory,
			splits=arguments.splits,
			samples_per_difficulty=arguments.samples_per_difficulty,
			allow_test_oracle=arguments.allow_test_oracle,
		)
	validation_summary_path = output_root / "label_validation_summary.json"
	if not (arguments.resume and validation_summary_path.is_file()):
		subprocess.run(
			build_validation_command(
				python_executable=Path(sys.executable),
				repository_root=repository_root,
				output_root=output_root,
			),
			cwd=repository_root,
			check=True,
		)
	print(json.dumps({"status": "complete", "output_root": str(output_root)}), flush=True)
	return 0


def _validate_idle_a800_gpus(
	gpu_ids: Sequence[int],
) -> Tuple[Dict[str, str], ...]:
	"""Require only the selected physical A800 GPUs to be completely idle."""
	if not gpu_ids or any(gpu_id not in GPU_IDS for gpu_id in gpu_ids):
		raise ValueError(f"gpu_ids must be a non-empty subset of {GPU_IDS}")
	query = subprocess.check_output(
		[
			"nvidia-smi",
			"--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
			"utilization.gpu",
			"--format=csv,noheader,nounits",
		],
		text=True,
	)
	rows = []
	for line in query.splitlines():
		index, uuid, name, total, used, free, utilization = [
			item.strip() for item in line.split(",")
		]
		rows.append(
			{
				"index": index,
				"uuid": uuid,
				"name": name,
				"memory_total_mib": total,
				"memory_used_mib": used,
				"memory_free_mib": free,
				"utilization_percent": utilization,
			},
		)
	if len(rows) != 2 or any("A800-SXM4-80GB" not in row["name"] for row in rows):
		raise RuntimeError(f"expected exactly two A800-SXM4-80GB GPUs, found {rows}")
	process_output = subprocess.run(
		[
			"nvidia-smi",
			"--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
			"--format=csv,noheader",
		],
		text=True,
		capture_output=True,
		check=True,
	).stdout.strip()
	processes = process_output.splitlines()
	selected_rows = []
	for gpu_id in gpu_ids:
		target = next(
			(row for row in rows if int(row["index"]) == gpu_id),
			None,
		)
		if target is None:
			raise RuntimeError(f"physical GPU {gpu_id} is unavailable")
		target_processes = [
			line
			for line in processes
			if line.startswith(target["uuid"] + ",")
		]
		if target_processes:
			raise RuntimeError(
				f"refusing to launch while physical GPU {gpu_id} has compute "
				f"processes: {target_processes}",
			)
		if int(target["memory_used_mib"]) > 64:
			raise RuntimeError(
				f"refusing to launch while physical GPU {gpu_id} uses "
				f"{target['memory_used_mib']} MiB",
			)
		selected_rows.append(target)
	return tuple(selected_rows)


def _validate_public_data(
	data_directory: Path,
	splits: Sequence[str],
	expected_samples_per_difficulty: int,
) -> Dict[str, Dict[str, object]]:
	manifest = {}
	all_question_ids = set()
	for difficulty in range(1, 6):
		path = data_directory / f"diff-{difficulty}.json"
		data_bytes = path.read_bytes()
		payload = json.loads(data_bytes)
		split_counts = {split: len(payload[split]) for split in splits}
		count = sum(split_counts.values())
		if count != expected_samples_per_difficulty:
			raise RuntimeError(
				f"unexpected selected split count for {path}: "
				f"{count} != {expected_samples_per_difficulty}",
			)
		for split in splits:
			for record in payload[split]:
				question_id = str(record.get("query_id", ""))
				if not question_id or question_id in all_question_ids:
					raise RuntimeError(
						f"empty or duplicate selected query_id: {question_id!r}",
					)
				all_question_ids.add(question_id)
		manifest[str(difficulty)] = {
			"path": str(path),
			"sha256": hashlib.sha256(data_bytes).hexdigest(),
			"selected_split_counts": split_counts,
			"selected_count": count,
		}
	return manifest


def _write_or_validate_manifest(
	output_root: Path,
	manifest: Dict[str, object],
	resume: bool,
) -> None:
	manifest_path = output_root / "run_manifest.json"
	if resume:
		stored = json.loads(manifest_path.read_text(encoding="utf-8"))
		for key in (
			"code_commit",
			"model_revision",
			"batch_size",
			"seed",
			"n_simulations",
			"reward_evaluator",
			"lossless_optimizations",
			"dataset_manifest",
			"data_files",
		):
			if stored.get(key) != manifest.get(key):
				raise RuntimeError(f"resume manifest mismatch for {key}")
		return
	output_root.mkdir(parents=True, exist_ok=False)
	with manifest_path.open("x", encoding="utf-8") as destination:
		json.dump(manifest, destination, indent=2, sort_keys=True)
		destination.write("\n")


def _run_gpu_shard(
	gpu_id: int,
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
	model_revision: str,
	batch_size: int,
	data_directory: Path,
	splits: Sequence[str],
	samples_per_difficulty: int,
	allow_test_oracle: bool,
) -> None:
	environment = dict(os.environ)
	environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	log_path = output_root / "logs" / f"gpu-{gpu_id}.log"
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("a", encoding="utf-8") as log:
		for difficulty in range(1, 6):
			command = build_label_command(
				python_executable=python_executable,
				repository_root=repository_root,
				output_root=output_root,
				model_revision=model_revision,
				batch_size=batch_size,
				difficulty=difficulty,
				shard_index=gpu_id,
				data_directory=data_directory,
				splits=splits,
				samples_per_difficulty=samples_per_difficulty,
				allow_test_oracle=allow_test_oracle,
			)
			log.write(json.dumps({"command": command}) + "\n")
			log.flush()
			subprocess.run(
				command,
				cwd=repository_root,
				env=environment,
				stdout=log,
				stderr=subprocess.STDOUT,
				check=True,
			)


def _merge_difficulty(
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
	difficulty: int,
	resume: bool,
	data_directory: Path,
	splits: Sequence[str],
	samples_per_difficulty: int,
	allow_test_oracle: bool,
) -> None:
	merged_path = (
		output_root
		/ "supervision"
		/ "meta-llama"
		/ "Llama-3.2-3B-Instruct"
		/ f"dart-math-diff-{difficulty}"
		/ "merged_mcts_samples.json"
	)
	if resume and merged_path.is_file():
		return
	command = (
		str(python_executable),
		str(repository_root / "scripts" / "merge_mcts_shards.py"),
		"--data-file",
		str(data_directory / f"diff-{difficulty}.json"),
		"--splits",
		*splits,
		"--trace-jsonl",
		str(output_root / f"diff-{difficulty}" / "trace-shard-0-of-2.jsonl"),
		str(output_root / f"diff-{difficulty}" / "trace-shard-1-of-2.jsonl"),
		"--output-json",
		str(merged_path),
		"--max-samples",
		str(samples_per_difficulty),
	)
	if allow_test_oracle:
		command += ("--allow-test-oracle",)
	subprocess.run(command, cwd=repository_root, check=True)


if __name__ == "__main__":
	raise SystemExit(main())
