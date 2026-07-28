#!/usr/bin/env python3
"""Launch the V2 two-GPU MCTS run once both assigned GPUs stay idle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.wait_for_idle_gpu_and_launch import (
	GpuState,
	_acquire_lock,
	_emit_status,
	_record_trigger_result,
	is_safely_idle,
	query_gpu_state,
)


GPU_IDS = (0, 1)


def all_gpus_are_safely_idle(
	states: Sequence[GpuState],
	max_memory_used_mib: int,
	max_utilization_percent: int,
) -> bool:
	"""Require both assigned physical GPUs to satisfy every idle condition."""
	if tuple(sorted(state.index for state in states)) != GPU_IDS:
		return False
	return all(
		is_safely_idle(
			state,
			max_memory_used_mib=max_memory_used_mib,
			max_utilization_percent=max_utilization_percent,
		)
		for state in states
	)


def wait_for_all_gpus_sustained_idle(
	query_states: Callable[[], Tuple[GpuState, ...]],
	required_idle_checks: int,
	poll_seconds: float,
	max_memory_used_mib: int,
	max_utilization_percent: int,
	sleep: Callable[[float], None] = time.sleep,
	status_callback: Optional[Callable[[dict], None]] = None,
) -> Tuple[GpuState, ...]:
	"""Wait for consecutive samples where both GPUs are simultaneously idle."""
	if required_idle_checks <= 0:
		raise ValueError("required_idle_checks must be positive")
	if poll_seconds <= 0:
		raise ValueError("poll_seconds must be positive")
	consecutive_idle_checks = 0
	while True:
		states = query_states()
		idle = all_gpus_are_safely_idle(
			states,
			max_memory_used_mib=max_memory_used_mib,
			max_utilization_percent=max_utilization_percent,
		)
		consecutive_idle_checks = consecutive_idle_checks + 1 if idle else 0
		if status_callback is not None:
			status_callback(
				{
					"event": "all_gpu_sample",
					"idle": idle,
					"consecutive_idle_checks": consecutive_idle_checks,
					"required_idle_checks": required_idle_checks,
					"states": [asdict(state) for state in states],
				},
			)
		if consecutive_idle_checks >= required_idle_checks:
			return states
		sleep(poll_seconds)


def build_v2_command(
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
	model_revision: str,
	batch_size: int,
	data_directory: Path,
) -> Tuple[str, ...]:
	"""Build the fixed 500-question, all-split, two-GPU command."""
	return (
		str(python_executable),
		str(repository_root / "scripts" / "run_mcts_labels_2gpu.py"),
		"--output-root",
		str(output_root),
		"--model-revision",
		model_revision,
		"--batch-size",
		str(batch_size),
		"--data-directory",
		str(data_directory),
		"--samples-per-difficulty",
		"100",
		"--splits",
		"train",
		"validation",
		"test",
		"--allow-test-oracle",
	)


def claim_v2_launch(state_path: Path, command: Sequence[str]) -> None:
	"""Create the permanent one-shot launch claim before starting the run."""
	state_path = Path(state_path)
	state_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"status": "launch_attempted",
		"gpu_indices": list(GPU_IDS),
		"command": tuple(command),
		"claimed_at_utc": datetime.now(timezone.utc).isoformat(),
	}
	try:
		with state_path.open("x", encoding="utf-8") as destination:
			json.dump(payload, destination, indent=2, sort_keys=True)
			destination.write("\n")
			destination.flush()
			os.fsync(destination.fileno())
	except FileExistsError:
		raise RuntimeError(
			f"one-shot V2 launch was already claimed according to {state_path}",
		) from None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--model-revision", required=True)
	parser.add_argument("--batch-size", type=int, default=50)
	parser.add_argument("--hf-home", type=Path, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	parser.add_argument(
		"--data-directory",
		type=Path,
		default=REPOSITORY_ROOT / "data" / "redm-public-v2",
	)
	parser.add_argument("--required-idle-checks", type=int, default=10)
	parser.add_argument("--poll-seconds", type=float, default=30.0)
	parser.add_argument("--max-memory-used-mib", type=int, default=64)
	parser.add_argument("--max-utilization-percent", type=int, default=5)
	parser.add_argument("--lock-path", type=Path, required=True)
	parser.add_argument("--state-path", type=Path, required=True)
	arguments = parser.parse_args(argv)
	if arguments.batch_size <= 0:
		parser.error("--batch-size must be positive")
	if arguments.required_idle_checks <= 0:
		parser.error("--required-idle-checks must be positive")
	if arguments.poll_seconds <= 0:
		parser.error("--poll-seconds must be positive")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	repository_root = arguments.repository_root.resolve()
	output_root = arguments.output_root.resolve()
	if output_root.exists():
		raise FileExistsError(f"V2 output root already exists: {output_root}")
	python_executable = repository_root / ".venv" / "bin" / "python"
	command = build_v2_command(
		python_executable=python_executable,
		repository_root=repository_root,
		output_root=output_root,
		model_revision=arguments.model_revision,
		batch_size=arguments.batch_size,
		data_directory=arguments.data_directory.resolve(),
	)
	environment = dict(os.environ)
	environment.update(
		{
			"HF_HOME": str(arguments.hf_home.resolve()),
			"HF_HUB_OFFLINE": "1",
			"TRANSFORMERS_OFFLINE": "1",
			"HF_DATASETS_OFFLINE": "1",
			"PYTHONUNBUFFERED": "1",
		},
	)
	lock_stream = _acquire_lock(arguments.lock_path.resolve())
	try:
		state_path = arguments.state_path.resolve()
		if state_path.exists():
			raise RuntimeError(
				f"one-shot V2 launch was already claimed according to {state_path}",
			)
		wait_for_all_gpus_sustained_idle(
			query_states=lambda: tuple(query_gpu_state(index) for index in GPU_IDS),
			required_idle_checks=arguments.required_idle_checks,
			poll_seconds=arguments.poll_seconds,
			max_memory_used_mib=arguments.max_memory_used_mib,
			max_utilization_percent=arguments.max_utilization_percent,
			status_callback=_emit_status,
		)
		claim_v2_launch(state_path, command)
		_emit_status({"event": "v2_one_shot_launch_attempt", "command": command})
		result = subprocess.run(
			command,
			cwd=repository_root,
			env=environment,
			check=False,
		)
		_record_trigger_result(state_path, result.returncode)
		_emit_status(
			{
				"event": "v2_one_shot_launch_finished",
				"returncode": result.returncode,
			},
		)
		return result.returncode
	finally:
		lock_stream.close()


if __name__ == "__main__":
	raise SystemExit(main())
