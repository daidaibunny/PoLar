#!/usr/bin/env python3
"""Run all five public Predictor-compatible MCTS label shards on two CUDA GPUs."""

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


def build_label_command(
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
	model_revision: str,
	batch_size: int,
	difficulty: int,
	shard_index: int,
) -> Tuple[str, ...]:
	"""Build one fixed-parameter command for a difficulty and physical GPU shard."""
	difficulty_root = output_root / f"diff-{difficulty}"
	return (
		str(python_executable),
		str(repository_root / "scripts" / "generate_mcts_labels.py"),
		"--data-file",
		str(repository_root / "data" / "redm-public" / f"diff-{difficulty}.json"),
		"--splits",
		"train",
		"validation",
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
		str(EXPECTED_SAMPLES_PER_DIFFICULTY),
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--model-revision", required=True)
	parser.add_argument("--batch-size", type=int, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	parser.add_argument("--resume", action="store_true")
	arguments = parser.parse_args(argv)
	if not re.fullmatch(r"[0-9a-f]{40}", arguments.model_revision):
		parser.error("--model-revision must be a 40-character commit SHA")
	if arguments.batch_size <= 0:
		parser.error("--batch-size must be positive")
	if arguments.output_root.exists() and not arguments.resume:
		parser.error("--output-root already exists; use a new path or --resume")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	repository_root = arguments.repository_root.resolve()
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
	gpu_state = _validate_idle_a800_pair()
	data_manifest = _validate_public_data(repository_root)
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
		"gpu_state_before_launch": gpu_state,
		"batch_size": arguments.batch_size,
		"seed": 42,
		"n_simulations": 200,
		"max_new_tokens": 50,
		"generation": "greedy",
		"ucb_c": 2**0.5,
		"length_penalty_lambda": 5.0,
		"random_action_probability": 0.1,
		"max_block_length": 4,
		"max_repeat_count_predictor": 1,
		"lossless_optimizations": [
			"reuse_exact_batch_token_tensors",
			"reuse_prompt_hashes",
			"reuse_official_math_evaluator",
			"memoize_identical_scoring_inputs",
			"reuse_flush_per_record_cache_stream",
			"skip_network_lookup_for_pinned_revision",
		],
		"splits": ["train", "validation"],
		"samples_per_difficulty": EXPECTED_SAMPLES_PER_DIFFICULTY,
		"data_files": data_manifest,
	}
	_write_or_validate_manifest(output_root, manifest, arguments.resume)

	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		futures = [
			executor.submit(
				_run_gpu_shard,
				gpu_id=gpu_id,
				python_executable=Path(sys.executable),
				repository_root=repository_root,
				output_root=output_root,
				model_revision=arguments.model_revision,
				batch_size=arguments.batch_size,
			)
			for gpu_id in GPU_IDS
		]
		for future in futures:
			future.result()

	for difficulty in range(1, 6):
		_merge_difficulty(
			python_executable=Path(sys.executable),
			repository_root=repository_root,
			output_root=output_root,
			difficulty=difficulty,
			resume=arguments.resume,
		)
	print(json.dumps({"status": "complete", "output_root": str(output_root)}), flush=True)
	return 0


def _validate_idle_a800_pair() -> Tuple[Dict[str, str], ...]:
	query = subprocess.check_output(
		[
			"nvidia-smi",
			"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
			"--format=csv,noheader,nounits",
		],
		text=True,
	)
	rows = []
	for line in query.splitlines():
		index, name, total, used, free, utilization = [item.strip() for item in line.split(",")]
		rows.append(
			{
				"index": index,
				"name": name,
				"memory_total_mib": total,
				"memory_used_mib": used,
				"memory_free_mib": free,
				"utilization_percent": utilization,
			},
		)
	if len(rows) != 2 or any("A800-SXM4-80GB" not in row["name"] for row in rows):
		raise RuntimeError(f"expected exactly two A800-SXM4-80GB GPUs, found {rows}")
	processes = subprocess.run(
		[
			"nvidia-smi",
			"--query-compute-apps=pid,process_name,used_gpu_memory",
			"--format=csv,noheader",
		],
		text=True,
		capture_output=True,
		check=True,
	).stdout.strip()
	if processes:
		raise RuntimeError(f"refusing to launch while GPU compute processes exist: {processes}")
	return tuple(rows)


def _validate_public_data(repository_root: Path) -> Dict[str, Dict[str, object]]:
	manifest = {}
	for difficulty in range(1, 6):
		path = repository_root / "data" / "redm-public" / f"diff-{difficulty}.json"
		data_bytes = path.read_bytes()
		payload = json.loads(data_bytes)
		count = len(payload["train"]) + len(payload["validation"])
		if count != EXPECTED_SAMPLES_PER_DIFFICULTY:
			raise RuntimeError(f"unexpected train+validation count for {path}: {count}")
		manifest[str(difficulty)] = {
			"path": str(path),
			"sha256": hashlib.sha256(data_bytes).hexdigest(),
			"train_validation_count": count,
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
			"lossless_optimizations",
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
		str(repository_root / "data" / "redm-public" / f"diff-{difficulty}.json"),
		"--splits",
		"train",
		"validation",
		"--trace-jsonl",
		str(output_root / f"diff-{difficulty}" / "trace-shard-0-of-2.jsonl"),
		str(output_root / f"diff-{difficulty}" / "trace-shard-1-of-2.jsonl"),
		"--output-json",
		str(merged_path),
		"--max-samples",
		str(EXPECTED_SAMPLES_PER_DIFFICULTY),
	)
	subprocess.run(command, cwd=repository_root, check=True)


if __name__ == "__main__":
	raise SystemExit(main())
