import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from scripts.wait_for_idle_gpu_and_launch import (
	GpuState,
	build_resume_command,
	claim_one_shot_trigger,
	is_safely_idle,
	wait_for_sustained_idle,
)


class WaitForIdleGpuAndLaunchTest(unittest.TestCase):
	def test_idle_requires_no_processes_low_memory_and_low_utilization(self) -> None:
		idle = GpuState(
			index=1,
			uuid="gpu-1",
			memory_used_mib=2,
			utilization_percent=0,
			compute_process_count=0,
		)
		busy_process = GpuState(1, "gpu-1", 2, 0, 1)
		busy_memory = GpuState(1, "gpu-1", 65, 0, 0)
		busy_utilization = GpuState(1, "gpu-1", 2, 6, 0)

		self.assertTrue(is_safely_idle(idle, 64, 5))
		self.assertFalse(is_safely_idle(busy_process, 64, 5))
		self.assertFalse(is_safely_idle(busy_memory, 64, 5))
		self.assertFalse(is_safely_idle(busy_utilization, 64, 5))

	def test_sustained_idle_counter_resets_after_a_busy_sample(self) -> None:
		idle = GpuState(1, "gpu-1", 2, 0, 0)
		busy = GpuState(1, "gpu-1", 18000, 0, 1)
		query_state = Mock(side_effect=[idle, idle, busy, idle, idle, idle])
		sleep = Mock()

		result = wait_for_sustained_idle(
			query_state=query_state,
			required_idle_checks=3,
			poll_seconds=30,
			max_memory_used_mib=64,
			max_utilization_percent=5,
			sleep=sleep,
		)

		self.assertEqual(result, idle)
		self.assertEqual(query_state.call_count, 6)
		self.assertEqual(sleep.call_count, 5)

	def test_builds_resume_command_for_only_selected_shard(self) -> None:
		command = build_resume_command(
			python_executable=Path("/repo/.venv/bin/python"),
			repository_root=Path("/repo"),
			output_root=Path("/output"),
			model_revision="a" * 40,
			batch_size=192,
			gpu_index=1,
		)

		self.assertEqual(command[0], "/repo/.venv/bin/python")
		self.assertIn("--only-shard", command)
		self.assertEqual(command[command.index("--only-shard") + 1], "1")
		self.assertIn("--resume", command)

	def test_one_shot_trigger_cannot_be_claimed_twice(self) -> None:
		with TemporaryDirectory() as directory:
			state_path = Path(directory) / "gpu-1-trigger.json"
			claim_one_shot_trigger(state_path, ("python", "run.py"), gpu_index=1)

			with self.assertRaises(RuntimeError):
				claim_one_shot_trigger(state_path, ("python", "run.py"), gpu_index=1)

			self.assertIn('"status": "launch_attempted"', state_path.read_text())


if __name__ == "__main__":
	unittest.main()
