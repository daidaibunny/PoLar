#!/usr/bin/env python3
"""Train five public PoLar Predictors on two CUDA GPUs with the README config."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
TRAINING_ASSIGNMENTS = {0: (1, 2), 1: (3, 4, 5)}


def resolve_cached_model_revision(hf_home: Path, model_id: str) -> str:
	"""Resolve and validate the immutable snapshot selected by a cache ref."""
	model_cache_name = "models--" + model_id.replace("/", "--")
	model_root = Path(hf_home) / "hub" / model_cache_name
	ref_path = model_root / "refs" / "main"
	if not ref_path.is_file():
		raise RuntimeError(f"cached model ref is missing: {ref_path}")
	revision = ref_path.read_text(encoding="utf-8").strip()
	if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
		raise RuntimeError(f"cached model ref is not a commit SHA: {ref_path}")
	snapshot_path = model_root / "snapshots" / revision
	if not snapshot_path.is_dir():
		raise RuntimeError(f"cached model snapshot is missing: {snapshot_path}")
	return revision


def resolve_cached_embedding_revision(hf_home: Path) -> str:
	"""Resolve the official frozen Predictor embedding model revision."""
	return resolve_cached_model_revision(hf_home, EMBEDDING_MODEL_ID)


def build_training_command(
	python_executable: Path,
	repository_root: Path,
	supervision_root: Path,
	output_root: Path,
	hf_cache_dir: Path,
	difficulty: int,
) -> Tuple[str, ...]:
	"""Build the official README starting command with recorded public splits."""
	return (
		str(python_executable),
		str(repository_root / "run_polar.py"),
		"--policy_mode",
		"polar",
		"--target_diff",
		str(difficulty),
		"--model_path",
		MODEL_ID,
		"--data_root",
		str(supervision_root),
		"--save_dir",
		str(output_root / f"diff-{difficulty}"),
		"--hf_cache_dir",
		str(hf_cache_dir),
		"--num_epochs",
		"10",
		"--batch_size",
		"128",
		"--learning_rate",
		"0.0005",
		"--max_paths_per_sample",
		"50",
		"--per_sample_weight_normalize",
		"--beam_size",
		"5",
		"--top_k_paths",
		"5",
		"--reweight_original_path_if_shorter_valid",
		"--original_path_weight",
		"0.30",
		"--seed",
		"42",
		"--lr_scheduler",
		"cosine",
		"--warmup_steps",
		"10",
		"--use_recorded_data_splits",
		"--train_only",
	)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--supervision-root", type=Path, required=True)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	arguments = parser.parse_args(argv)
	if arguments.output_root.exists():
		parser.error("--output-root already exists; use a unique training output")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	repository_root = arguments.repository_root.resolve()
	supervision_root = arguments.supervision_root.resolve()
	output_root = arguments.output_root.resolve()
	hf_home_value = os.environ.get("HF_HOME")
	if not hf_home_value:
		raise RuntimeError("HF_HOME must point to the shared Hugging Face cache")
	hf_home = Path(hf_home_value).resolve()
	if not hf_home.is_dir():
		raise RuntimeError(f"HF_HOME does not exist: {hf_home}")
	embedding_model_revision = resolve_cached_embedding_revision(hf_home)
	_validate_supervision(supervision_root)
	from scripts.run_mcts_labels_2gpu import _validate_idle_a800_gpus

	gpu_state = _validate_idle_a800_gpus((0, 1))
	commit = subprocess.check_output(
		["git", "rev-parse", "HEAD"],
		cwd=repository_root,
		text=True,
	).strip()
	output_root.mkdir(parents=True, exist_ok=False)
	manifest = {
		"method_claim": "independent reconstruction",
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"code_commit": commit,
		"model_id_for_depth_and_data": MODEL_ID,
		"predictor_embedding_model": EMBEDDING_MODEL_ID,
		"predictor_embedding_model_revision": embedding_model_revision,
		"parallelism": "five independent Predictors scheduled over two CUDA GPUs",
		"physical_gpu_0_difficulties": list(TRAINING_ASSIGNMENTS[0]),
		"physical_gpu_1_difficulties": list(TRAINING_ASSIGNMENTS[1]),
		"gpu_state_before_launch": gpu_state,
		"num_epochs": 10,
		"batch_size": 128,
		"learning_rate": 5e-4,
		"max_paths_per_sample": 50,
		"per_sample_weight_normalize": True,
		"beam_size": 5,
		"top_k_paths": 5,
		"reweight_original_path_if_shorter_valid": True,
		"original_path_weight": 0.30,
		"seed": 42,
		"lr_scheduler": "cosine",
		"warmup_steps": 10,
		"split_definition": "search_metadata.data_split",
	}
	with (output_root / "run_manifest.json").open("x", encoding="utf-8") as destination:
		json.dump(manifest, destination, indent=2, sort_keys=True)
		destination.write("\n")

	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		futures = [
			executor.submit(
				_run_training_shard,
				gpu_id=gpu_id,
				difficulties=difficulties,
				python_executable=Path(sys.executable),
				repository_root=repository_root,
				supervision_root=supervision_root,
				output_root=output_root,
				hf_home=hf_home,
			)
			for gpu_id, difficulties in TRAINING_ASSIGNMENTS.items()
		]
		for future in futures:
			future.result()
	print(json.dumps({"status": "complete", "output_root": str(output_root)}), flush=True)
	return 0


def _validate_supervision(supervision_root: Path) -> None:
	model_root = supervision_root / "meta-llama" / "Llama-3.2-3B-Instruct"
	for difficulty in range(1, 6):
		path = model_root / f"dart-math-diff-{difficulty}" / "merged_mcts_samples.json"
		payload = json.loads(path.read_text(encoding="utf-8"))
		samples = payload["samples"]
		counts = {"train": 0, "validation": 0}
		for sample in samples:
			split = sample.get("search_metadata", {}).get("data_split")
			if split in counts:
				counts[split] += 1
		if counts != {"train": 938, "validation": 187}:
			raise RuntimeError(f"unexpected recorded split counts for {path}: {counts}")


def _run_training_shard(
	gpu_id: int,
	difficulties: Sequence[int],
	python_executable: Path,
	repository_root: Path,
	supervision_root: Path,
	output_root: Path,
	hf_home: Path,
) -> None:
	environment = dict(os.environ)
	environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	environment["HF_HUB_OFFLINE"] = "1"
	environment["TRANSFORMERS_OFFLINE"] = "1"
	environment["HF_DATASETS_OFFLINE"] = "1"
	environment["PYTHONHASHSEED"] = "42"
	log_path = output_root / "logs" / f"gpu-{gpu_id}.log"
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("a", encoding="utf-8") as log:
		for difficulty in difficulties:
			command = build_training_command(
				python_executable=python_executable,
				repository_root=repository_root,
				supervision_root=supervision_root,
				output_root=output_root,
				hf_cache_dir=hf_home,
				difficulty=difficulty,
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


if __name__ == "__main__":
	raise SystemExit(main())
