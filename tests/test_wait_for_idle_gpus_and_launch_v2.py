import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.wait_for_idle_gpu_and_launch import GpuState
from scripts.wait_for_idle_gpus_and_launch_v2 import (
	all_gpus_are_safely_idle,
	build_v2_command,
	wait_for_all_gpus_sustained_idle,
)


class WaitForIdleGpusAndLaunchV2Test(unittest.TestCase):
	def test_all_gpus_must_be_idle(self) -> None:
		idle_zero = GpuState(0, "gpu-0", 2, 0, 0)
		idle_one = GpuState(1, "gpu-1", 2, 0, 0)
		busy_one = GpuState(1, "gpu-1", 20000, 100, 1)

		self.assertTrue(all_gpus_are_safely_idle((idle_zero, idle_one), 64, 5))
		self.assertFalse(all_gpus_are_safely_idle((idle_zero, busy_one), 64, 5))

	def test_sustained_counter_resets_when_either_gpu_becomes_busy(self) -> None:
		idle = (
			GpuState(0, "gpu-0", 2, 0, 0),
			GpuState(1, "gpu-1", 2, 0, 0),
		)
		busy = (
			GpuState(0, "gpu-0", 2, 0, 0),
			GpuState(1, "gpu-1", 20000, 100, 1),
		)
		query_states = Mock(side_effect=[idle, idle, busy, idle, idle, idle])
		sleep = Mock()

		result = wait_for_all_gpus_sustained_idle(
			query_states=query_states,
			required_idle_checks=3,
			poll_seconds=30,
			max_memory_used_mib=64,
			max_utilization_percent=5,
			sleep=sleep,
		)

		self.assertEqual(result, idle)
		self.assertEqual(query_states.call_count, 6)

	def test_builds_v2_all_split_two_gpu_command(self) -> None:
		command = build_v2_command(
			python_executable=Path("/repo/.venv/bin/python"),
			repository_root=Path("/repo"),
			output_root=Path("/output"),
			model_revision="a" * 40,
			batch_size=50,
			data_directory=Path("/repo/data/redm-public-v2"),
		)

		joined = " ".join(command)
		self.assertIn("--splits train validation test", joined)
		self.assertIn("--samples-per-difficulty 100", joined)
		self.assertIn("--allow-test-oracle", joined)
		self.assertNotIn("--only-shard", joined)
		self.assertNotIn("--resume", joined)


if __name__ == "__main__":
	unittest.main()
