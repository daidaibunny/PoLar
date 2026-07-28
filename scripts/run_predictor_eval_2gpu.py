#!/usr/bin/env python3
"""Evaluate five public PoLar Predictors online across two CUDA GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EVALUATION_ASSIGNMENTS = {0: (1, 3, 5), 1: (2, 4)}


def build_evaluation_command(
	python_executable: Path,
	repository_root: Path,
	predictor_data_root: Path,
	output_root: Path,
	hf_cache_dir: Path,
	checkpoint_path: Path,
	difficulty: int,
) -> Tuple[str, ...]:
	"""Build official top-five beam evaluation with forced online execution."""
	return (
		str(python_executable),
		str(repository_root / "run_polar.py"),
		"--eval",
		"--policy_mode",
		"polar",
		"--target_diff",
		str(difficulty),
		"--model_path",
		MODEL_ID,
		"--data_root",
		str(predictor_data_root),
		"--save_dir",
		str(output_root / f"diff-{difficulty}"),
		"--hf_cache_dir",
		str(hf_cache_dir),
		"--checkpoint_path",
		str(checkpoint_path),
		"--num_epochs",
		"10",
		"--batch_size",
		"128",
		"--learning_rate",
		"0.0005",
		"--max_paths_per_sample",
		"50",
		"--per_sample_weight_normalize",
		"--reweight_original_path_if_shorter_valid",
		"--original_path_weight",
		"0.30",
		"--beam_size",
		"5",
		"--top_k_ops",
		"2",
		"--top_k_paths",
		"5",
		"--seg_threshold",
		"0.5",
		"--max_new_tokens",
		"50",
		"--num_samples",
		"375",
		"--seed",
		"42",
		"--lr_scheduler",
		"cosine",
		"--warmup_steps",
		"10",
		"--use_recorded_data_splits",
		"--no_trust_valid_cache",
		"--eval_cache_breakdown",
	)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--predictor-data-root", type=Path, required=True)
	parser.add_argument("--training-output-root", type=Path, required=True)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	arguments = parser.parse_args(argv)
	if arguments.output_root.exists():
		parser.error("--output-root already exists; use a unique evaluation output")
	return arguments


def _find_checkpoints(training_output_root: Path) -> Dict[int, Path]:
	checkpoints = {}
	for difficulty in range(1, 6):
		matches = tuple((training_output_root / f"diff-{difficulty}").rglob("*.pt"))
		if len(matches) != 1:
			raise RuntimeError(
				f"expected one checkpoint for difficulty {difficulty}, found {matches}",
			)
		checkpoints[difficulty] = matches[0].resolve()
	return checkpoints


def _validate_predictor_data(predictor_data_root: Path) -> None:
	model_root = predictor_data_root / "meta-llama" / "Llama-3.2-3B-Instruct"
	for difficulty in range(1, 6):
		path = model_root / f"dart-math-diff-{difficulty}" / "merged_mcts_samples.json"
		samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
		counts = {"train": 0, "validation": 0, "test": 0}
		for sample in samples:
			split = sample.get("search_metadata", {}).get("data_split")
			if split in counts:
				counts[split] += 1
			if split == "test" and (
				sample.get("final_valid_transitions")
				or sample.get("final_invalid_transitions")
			):
				raise RuntimeError(f"test oracle labels found in {path}")
		if counts != {"train": 938, "validation": 187, "test": 375}:
			raise RuntimeError(f"unexpected recorded split counts for {path}: {counts}")


def _run_evaluation_shard(
	gpu_id: int,
	difficulties: Sequence[int],
	python_executable: Path,
	repository_root: Path,
	predictor_data_root: Path,
	output_root: Path,
	hf_home: Path,
	checkpoints: Dict[int, Path],
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
			command = build_evaluation_command(
				python_executable=python_executable,
				repository_root=repository_root,
				predictor_data_root=predictor_data_root,
				output_root=output_root,
				hf_cache_dir=hf_home,
				checkpoint_path=checkpoints[difficulty],
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


def _summarize_pass_at_k(output_root: Path) -> Dict[str, object]:
	summary = {}
	for difficulty in range(1, 6):
		matches = tuple((output_root / f"diff-{difficulty}").rglob("eval_results_*.json"))
		if len(matches) != 1:
			raise RuntimeError(
				f"expected one evaluation result for difficulty {difficulty}, found {matches}",
			)
		rows = json.loads(matches[0].read_text(encoding="utf-8"))
		pass_at_k = {}
		for k in range(1, 6):
			correct = sum(
				any(float(path["score"]) == 1.0 for path in row["top_paths"][:k])
				for row in rows
			)
			pass_at_k[str(k)] = correct / max(1, len(rows))
		summary[str(difficulty)] = {
			"samples": len(rows),
			"pass_at_k": pass_at_k,
			"result_path": str(matches[0]),
		}
	return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	repository_root = arguments.repository_root.resolve()
	predictor_data_root = arguments.predictor_data_root.resolve()
	training_output_root = arguments.training_output_root.resolve()
	output_root = arguments.output_root.resolve()
	hf_home_value = os.environ.get("HF_HOME")
	if not hf_home_value:
		raise RuntimeError("HF_HOME must point to the shared Hugging Face cache")
	hf_home = Path(hf_home_value).resolve()
	from scripts.run_predictor_training_2gpu import (
		resolve_cached_embedding_revision,
		resolve_cached_model_revision,
	)

	embedding_model_revision = resolve_cached_embedding_revision(hf_home)
	base_model_revision = resolve_cached_model_revision(hf_home, MODEL_ID)
	_validate_predictor_data(predictor_data_root)
	checkpoints = _find_checkpoints(training_output_root)
	from scripts.run_mcts_labels_2gpu import _validate_idle_a800_gpus

	gpu_state = _validate_idle_a800_gpus((0, 1))
	commit = subprocess.check_output(
		["git", "rev-parse", "HEAD"],
		cwd=repository_root,
		text=True,
	).strip()
	output_root.mkdir(parents=True, exist_ok=False)
	manifest = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"code_commit": commit,
		"model_id": MODEL_ID,
		"base_model_revision": base_model_revision,
		"predictor_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
		"predictor_embedding_model_revision": embedding_model_revision,
		"evaluation": "held-out recorded test split with online path execution",
		"trust_valid_cache": False,
		"samples_per_difficulty": 375,
		"beam_size": 5,
		"top_k_paths": 5,
		"max_new_tokens": 50,
		"gpu_assignments": {
			str(gpu_id): list(difficulties)
			for gpu_id, difficulties in EVALUATION_ASSIGNMENTS.items()
		},
		"gpu_state_before_launch": gpu_state,
		"checkpoints": {
			str(difficulty): str(path)
			for difficulty, path in checkpoints.items()
		},
	}
	with (output_root / "run_manifest.json").open("x", encoding="utf-8") as destination:
		json.dump(manifest, destination, indent=2, sort_keys=True)
		destination.write("\n")

	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		futures = [
			executor.submit(
				_run_evaluation_shard,
				gpu_id=gpu_id,
				difficulties=difficulties,
				python_executable=Path(sys.executable),
				repository_root=repository_root,
				predictor_data_root=predictor_data_root,
				output_root=output_root,
				hf_home=hf_home,
				checkpoints=checkpoints,
			)
			for gpu_id, difficulties in EVALUATION_ASSIGNMENTS.items()
		]
		for future in futures:
			future.result()

	summary = _summarize_pass_at_k(output_root)
	with (output_root / "pass_at_k_summary.json").open("x", encoding="utf-8") as destination:
		json.dump(summary, destination, indent=2, sort_keys=True)
		destination.write("\n")
	print(json.dumps({"status": "complete", "output_root": str(output_root)}))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
