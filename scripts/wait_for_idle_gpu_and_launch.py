#!/usr/bin/env python3
"""Wait for sustained GPU idleness before resuming one MCTS label shard."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, TextIO, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GpuState:
	"""Safety-relevant state for one physical GPU."""

	index: int
	uuid: str
	memory_used_mib: int
	utilization_percent: int
	compute_process_count: int


def query_gpu_state(gpu_index: int) -> GpuState:
	"""Read utilization and UUID-scoped compute processes from NVIDIA SMI."""
	gpu_output = subprocess.check_output(
		[
			"nvidia-smi",
			"--query-gpu=index,uuid,memory.used,utilization.gpu",
			"--format=csv,noheader,nounits",
		],
		text=True,
	)
	gpu_rows = {}
	for line in gpu_output.splitlines():
		index, uuid, memory_used, utilization = [
			value.strip() for value in line.split(",")
		]
		gpu_rows[int(index)] = (uuid, int(memory_used), int(utilization))
	if gpu_index not in gpu_rows:
		raise RuntimeError(f"physical GPU {gpu_index} is unavailable")

	process_output = subprocess.run(
		[
			"nvidia-smi",
			"--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
			"--format=csv,noheader,nounits",
		],
		text=True,
		capture_output=True,
		check=True,
	).stdout
	uuid, memory_used, utilization = gpu_rows[gpu_index]
	process_count = sum(
		1
		for line in process_output.splitlines()
		if line.strip() and line.split(",", maxsplit=1)[0].strip() == uuid
	)
	return GpuState(
		index=gpu_index,
		uuid=uuid,
		memory_used_mib=memory_used,
		utilization_percent=utilization,
		compute_process_count=process_count,
	)


def is_safely_idle(
	state: GpuState,
	max_memory_used_mib: int,
	max_utilization_percent: int,
) -> bool:
	"""Return whether every configured idle-safety condition is satisfied."""
	return (
		state.compute_process_count == 0
		and state.memory_used_mib <= max_memory_used_mib
		and state.utilization_percent <= max_utilization_percent
	)


def wait_for_sustained_idle(
	query_state: Callable[[], GpuState],
	required_idle_checks: int,
	poll_seconds: float,
	max_memory_used_mib: int,
	max_utilization_percent: int,
	sleep: Callable[[float], None] = time.sleep,
	status_callback: Optional[Callable[[dict], None]] = None,
) -> GpuState:
	"""Wait until all idle conditions hold for the required consecutive checks."""
	if required_idle_checks <= 0:
		raise ValueError("required_idle_checks must be positive")
	if poll_seconds <= 0:
		raise ValueError("poll_seconds must be positive")
	consecutive_idle_checks = 0
	while True:
		state = query_state()
		idle = is_safely_idle(
			state,
			max_memory_used_mib=max_memory_used_mib,
			max_utilization_percent=max_utilization_percent,
		)
		consecutive_idle_checks = consecutive_idle_checks + 1 if idle else 0
		if status_callback is not None:
			status_callback(
				{
					"event": "gpu_sample",
					"idle": idle,
					"consecutive_idle_checks": consecutive_idle_checks,
					"required_idle_checks": required_idle_checks,
					"state": asdict(state),
				},
			)
		if consecutive_idle_checks >= required_idle_checks:
			return state
		sleep(poll_seconds)


def build_resume_command(
	python_executable: Path,
	repository_root: Path,
	output_root: Path,
	model_revision: str,
	batch_size: int,
	gpu_index: int,
) -> Tuple[str, ...]:
	"""Build the fixed resume command for one physical GPU and shard."""
	return (
		str(python_executable),
		str(repository_root / "scripts" / "run_mcts_labels_2gpu.py"),
		"--output-root",
		str(output_root),
		"--model-revision",
		model_revision,
		"--batch-size",
		str(batch_size),
		"--only-shard",
		str(gpu_index),
		"--resume",
	)


def claim_one_shot_trigger(
	state_path: Path,
	command: Sequence[str],
	gpu_index: int,
) -> None:
	"""Atomically claim the only permitted launch attempt for this monitor."""
	state_path = Path(state_path)
	state_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"status": "launch_attempted",
		"gpu_index": gpu_index,
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
			f"one-shot launch was already claimed according to {state_path}",
		) from None


def _record_trigger_result(state_path: Path, returncode: int) -> None:
	"""Record completion without removing the permanent one-shot claim."""
	payload = json.loads(state_path.read_text(encoding="utf-8"))
	payload["status"] = "completed" if returncode == 0 else "failed"
	payload["returncode"] = returncode
	payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
	temporary_path = state_path.with_name(f"{state_path.name}.tmp.{os.getpid()}")
	with temporary_path.open("x", encoding="utf-8") as destination:
		json.dump(payload, destination, indent=2, sort_keys=True)
		destination.write("\n")
		destination.flush()
		os.fsync(destination.fileno())
	os.replace(temporary_path, state_path)


def _emit_status(payload: dict) -> None:
	payload = {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		**payload,
	}
	print(json.dumps(payload, sort_keys=True), flush=True)


def _acquire_lock(lock_path: Path) -> TextIO:
	lock_path.parent.mkdir(parents=True, exist_ok=True)
	lock_stream = lock_path.open("a+", encoding="utf-8")
	try:
		fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
	except BlockingIOError:
		lock_stream.close()
		raise RuntimeError(f"another monitor already holds {lock_path}") from None
	return lock_stream


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--model-revision", required=True)
	parser.add_argument("--batch-size", type=int, required=True)
	parser.add_argument("--hf-home", type=Path, required=True)
	parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
	parser.add_argument("--gpu-index", type=int, choices=(0, 1), default=1)
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
	python_executable = repository_root / ".venv" / "bin" / "python"
	command = build_resume_command(
		python_executable=python_executable,
		repository_root=repository_root,
		output_root=arguments.output_root.resolve(),
		model_revision=arguments.model_revision,
		batch_size=arguments.batch_size,
		gpu_index=arguments.gpu_index,
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
				f"one-shot launch was already claimed according to {state_path}",
			)
		wait_for_sustained_idle(
			query_state=lambda: query_gpu_state(arguments.gpu_index),
			required_idle_checks=arguments.required_idle_checks,
			poll_seconds=arguments.poll_seconds,
			max_memory_used_mib=arguments.max_memory_used_mib,
			max_utilization_percent=arguments.max_utilization_percent,
			status_callback=_emit_status,
		)
		claim_one_shot_trigger(state_path, command, arguments.gpu_index)
		_emit_status({"event": "one_shot_launch_attempt", "command": command})
		result = subprocess.run(
			command,
			cwd=repository_root,
			env=environment,
			check=False,
		)
		_record_trigger_result(state_path, result.returncode)
		_emit_status(
			{
				"event": "one_shot_launch_finished",
				"gpu_index": arguments.gpu_index,
				"returncode": result.returncode,
			},
		)
		return result.returncode
	finally:
		lock_stream.close()


if __name__ == "__main__":
	sys.exit(main())
