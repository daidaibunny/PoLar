import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_mcts_labels_2gpu import (
	OFFICIAL_REWARD_EVALUATOR,
	_validate_idle_a800_gpus,
	build_label_command,
	build_validation_command,
	parse_args,
)


class TwoGpuLabelCommandTest(unittest.TestCase):
	def test_manifest_uses_official_polar_reward_evaluator(self) -> None:
		self.assertEqual(
			OFFICIAL_REWARD_EVALUATOR,
			{
				"implementation": "dart_math.eval.EvaluatorMathBatch",
				"strict_extract": True,
				"use_orig_eq_for_olympiadbench": True,
				"timeout_seconds": 60,
				"processes_per_gpu": 4,
				"query": "",
				"dataset": "math",
				"source": "official PoLar polar/eval.py::_batch_compare_answers",
			},
		)

	def test_builds_fixed_paper_and_public_code_configuration(self) -> None:
		command = build_label_command(
			python_executable=Path(".venv/bin/python"),
			repository_root=Path("/repo"),
			output_root=Path("/output"),
			model_revision="a" * 40,
			batch_size=32,
			difficulty=3,
			shard_index=1,
		)

		joined = " ".join(command)
		self.assertIn("--n-simulations 200", joined)
		self.assertIn("--length-penalty-lambda 5.0", joined)
		self.assertIn("--random-action-probability 0.1", joined)
		self.assertIn("--ucb-c 1.4142135623730951", joined)
		self.assertIn("--max-new-tokens 50", joined)
		self.assertIn("--seed 42", joined)
		self.assertIn("--mode predictor_compatible", joined)
		self.assertIn("--splits train validation", joined)
		self.assertIn("--num-shards 2 --shard-index 1", joined)
		self.assertIn("--max-samples 1125", joined)
		self.assertIn("--batch-size 32 --search-width 32", joined)

	def test_builds_v2_all_split_oracle_command(self) -> None:
		command = build_label_command(
			python_executable=Path(".venv/bin/python"),
			repository_root=Path("/repo"),
			output_root=Path("/output"),
			model_revision="a" * 40,
			batch_size=50,
			difficulty=4,
			shard_index=0,
			data_directory=Path("/repo/data/redm-public-v2"),
			splits=("train", "validation", "test"),
			samples_per_difficulty=100,
			allow_test_oracle=True,
		)

		joined = " ".join(command)
		self.assertIn("/repo/data/redm-public-v2/diff-4.json", joined)
		self.assertIn("--splits train validation test", joined)
		self.assertIn("--allow-test-oracle", joined)
		self.assertIn("--max-samples 100", joined)
		self.assertIn("--batch-size 50 --search-width 50", joined)

	def test_builds_all_difficulty_trace_validation_command(self) -> None:
		command = build_validation_command(
			python_executable=Path(".venv/bin/python"),
			repository_root=Path("/repo"),
			output_root=Path("/output"),
		)

		joined = " ".join(command)
		self.assertIn("/repo/scripts/validate_mcts_smoke.py", joined)
		for difficulty in range(1, 6):
			for shard in range(2):
				self.assertIn(
					f"/output/diff-{difficulty}/trace-shard-{shard}-of-2.jsonl",
					joined,
				)
		self.assertIn("--summary-json /output/label_validation_summary.json", joined)

	def test_accepts_one_fixed_shard_for_staggered_execution(self) -> None:
		arguments = parse_args(
			[
				"--output-root",
				"/new/output",
				"--model-revision",
				"a" * 40,
				"--batch-size",
				"192",
				"--only-shard",
				"0",
			],
		)

		self.assertEqual(arguments.only_shard, 0)

	def test_requires_explicit_oracle_flag_for_v2_test_labels(self) -> None:
		with self.assertRaises(SystemExit):
			parse_args(
				[
					"--output-root",
					"/new/output",
					"--model-revision",
					"a" * 40,
					"--batch-size",
					"50",
					"--splits",
					"train",
					"validation",
					"test",
				],
			)

	def test_accepts_explicit_v2_all_split_oracle_labels(self) -> None:
		arguments = parse_args(
			[
				"--output-root",
				"/new/output",
				"--model-revision",
				"a" * 40,
				"--batch-size",
				"50",
				"--data-directory",
				"/repo/data/redm-public-v2",
				"--samples-per-difficulty",
				"100",
				"--splits",
				"train",
				"validation",
				"test",
				"--allow-test-oracle",
			],
		)

		self.assertEqual(arguments.samples_per_difficulty, 100)
		self.assertEqual(arguments.splits, ["train", "validation", "test"])
		self.assertTrue(arguments.allow_test_oracle)

	@patch("scripts.run_mcts_labels_2gpu.subprocess.run")
	@patch("scripts.run_mcts_labels_2gpu.subprocess.check_output")
	def test_selected_idle_gpu_ignores_a_busy_unselected_gpu(
		self,
		check_output,
		run,
	) -> None:
		check_output.return_value = (
			"0, uuid-0, NVIDIA A800-SXM4-80GB, 81920, 2, 81918, 0\n"
			"1, uuid-1, NVIDIA A800-SXM4-80GB, 81920, 33000, 48920, 100\n"
		)
		run.return_value.stdout = "uuid-1, 456, python, 32990 MiB\n"

		state = _validate_idle_a800_gpus((0,))

		self.assertEqual(tuple(row["index"] for row in state), ("0",))

	@patch("scripts.run_mcts_labels_2gpu.subprocess.run")
	@patch("scripts.run_mcts_labels_2gpu.subprocess.check_output")
	def test_rejects_a_busy_selected_gpu(self, check_output, run) -> None:
		check_output.return_value = (
			"0, uuid-0, NVIDIA A800-SXM4-80GB, 81920, 33000, 48920, 100\n"
			"1, uuid-1, NVIDIA A800-SXM4-80GB, 81920, 2, 81918, 0\n"
		)
		run.return_value.stdout = "uuid-0, 123, python, 32990 MiB\n"

		with self.assertRaises(RuntimeError):
			_validate_idle_a800_gpus((0,))


if __name__ == "__main__":
	unittest.main()
