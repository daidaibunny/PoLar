import unittest

from scripts.merge_mcts_shards import parse_args


class MergeMctsShardsScriptTest(unittest.TestCase):
	def test_requires_explicit_oracle_flag_for_test_split(self) -> None:
		with self.assertRaises(SystemExit):
			parse_args(
				[
					"--data-file",
					"diff-1.json",
					"--splits",
					"train",
					"test",
					"--trace-jsonl",
					"trace.jsonl",
					"--output-json",
					"merged.json",
				],
			)

	def test_accepts_explicit_all_split_oracle_merge(self) -> None:
		arguments = parse_args(
			[
				"--data-file",
				"diff-1.json",
				"--splits",
				"train",
				"validation",
				"test",
				"--allow-test-oracle",
				"--trace-jsonl",
				"trace.jsonl",
				"--output-json",
				"merged.json",
				"--max-samples",
				"100",
			],
		)

		self.assertEqual(arguments.splits, ["train", "validation", "test"])
		self.assertTrue(arguments.allow_test_oracle)
		self.assertEqual(arguments.max_samples, 100)


if __name__ == "__main__":
	unittest.main()
