import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.resume_mcts_labels_v2_optimized import (
	EXPECTED_QUESTIONS_PER_SHARD,
	ShardTask,
	build_shard_tasks,
	parse_args,
	run_tasks,
)


class OptimizedV2ResumeTest(unittest.TestCase):
	def test_requires_the_existing_v2_batch_and_question_counts(self) -> None:
		base_arguments = [
			"--output-root",
			"/output",
			"--model-revision",
			"a" * 40,
			"--hf-home",
			"/hf",
			"--data-directory",
			"/data",
			"--expected-original-code-commit",
			"b" * 40,
		]
		arguments = parse_args(base_arguments)

		self.assertEqual(arguments.batch_size, 50)
		self.assertEqual(arguments.samples_per_difficulty, 100)
		with self.assertRaises(SystemExit):
			parse_args([*base_arguments, "--batch-size", "192"])

	def test_builds_isolated_tasks_and_skips_a_complete_trace(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			output_root = Path(directory) / "output"
			complete_trace = (
				output_root / "diff-1" / "trace-shard-0-of-2.jsonl"
			)
			complete_trace.parent.mkdir(parents=True)
			complete_trace.write_text(
				"".join(
					json.dumps({"question_id": f"q{index}"}) + "\n"
					for index in range(EXPECTED_QUESTIONS_PER_SHARD)
				),
				encoding="utf-8",
			)

			tasks = build_shard_tasks(
				python_executable=Path("/repo/.venv/bin/python"),
				repository_root=Path("/repo"),
				output_root=output_root,
				model_revision="a" * 40,
				batch_size=50,
				data_directory=Path("/repo/data/redm-public-v2"),
				samples_per_difficulty=100,
			)

		self.assertEqual(len(tasks[0]), 4)
		self.assertEqual(len(tasks[1]), 5)
		self.assertNotIn(1, [task.difficulty for task in tasks[0]])
		all_log_paths = [task.log_path for gpu_tasks in tasks.values() for task in gpu_tasks]
		self.assertEqual(len(all_log_paths), len(set(all_log_paths)))
		joined_command = " ".join(tasks[1][0].command)
		self.assertIn("--batch-size 50 --search-width 50", joined_command)
		self.assertIn("--n-simulations 200", joined_command)
		self.assertIn("--splits train validation test", joined_command)
		self.assertIn("--allow-test-oracle", joined_command)

	@patch("scripts.resume_mcts_labels_v2_optimized.run_task")
	def test_runs_exactly_two_workers_per_physical_gpu(self, run_task) -> None:
		barrier = threading.Barrier(4)
		lock = threading.Lock()
		active = {0: 0, 1: 0}
		maximum_active = {0: 0, 1: 0}

		def observe_concurrency(task, _repository_root, _environment):
			with lock:
				active[task.gpu_id] += 1
				maximum_active[task.gpu_id] = max(
					maximum_active[task.gpu_id],
					active[task.gpu_id],
				)
			barrier.wait(timeout=2)
			with lock:
				active[task.gpu_id] -= 1

		run_task.side_effect = observe_concurrency
		tasks = {
			gpu_id: tuple(
				ShardTask(
					gpu_id=gpu_id,
					difficulty=difficulty,
					trace_path=Path(f"trace-{gpu_id}-{difficulty}"),
					cache_path=Path(f"cache-{gpu_id}-{difficulty}"),
					log_path=Path(f"log-{gpu_id}-{difficulty}"),
					command=("python",),
				)
				for difficulty in (1, 2)
			)
			for gpu_id in (0, 1)
		}

		run_tasks(tasks, Path("/repo"), {})

		self.assertEqual(maximum_active, {0: 2, 1: 2})


if __name__ == "__main__":
	unittest.main()
