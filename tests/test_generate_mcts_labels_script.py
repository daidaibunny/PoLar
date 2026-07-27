import unittest
from pathlib import Path

from scripts.generate_mcts_labels import parse_args


class GenerateMctsLabelsCliTest(unittest.TestCase):
	def test_defaults_to_training_and_validation_predictor_search(self) -> None:
		arguments = parse_args(
			[
				"--data-file",
				"diff-1.json",
				"--output-jsonl",
				"records.jsonl",
				"--cache-jsonl",
				"cache.jsonl",
			],
		)

		self.assertEqual(arguments.data_file, Path("diff-1.json"))
		self.assertEqual(arguments.splits, ["train", "validation"])
		self.assertEqual(arguments.mode, "predictor_compatible")
		self.assertEqual(arguments.n_simulations, 20)

	def test_requires_explicit_oracle_flag_for_test_split(self) -> None:
		with self.assertRaises(SystemExit):
			parse_args(
				[
					"--data-file",
					"diff-1.json",
					"--output-jsonl",
					"records.jsonl",
					"--cache-jsonl",
					"cache.jsonl",
					"--splits",
					"test",
				],
			)

	def test_accepts_explicit_test_oracle_and_sharding(self) -> None:
		arguments = parse_args(
			[
				"--data-file",
				"diff-1.json",
				"--output-jsonl",
				"records.jsonl",
				"--cache-jsonl",
				"cache.jsonl",
				"--splits",
				"test",
				"--allow-test-oracle",
				"--num-shards",
				"2",
				"--shard-index",
				"1",
			],
		)

		self.assertTrue(arguments.allow_test_oracle)
		self.assertEqual(arguments.num_shards, 2)
		self.assertEqual(arguments.shard_index, 1)


if __name__ == "__main__":
	unittest.main()
