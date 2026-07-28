import unittest
from pathlib import Path

from scripts.validate_mcts_smoke import (
	classify_predictor_path,
	parse_args,
	smoke_failures,
	summarize_trace_records,
)


class PredictorPathClassificationTest(unittest.TestCase):
	def test_classifies_official_parser_paths_by_operation_mix(self) -> None:
		self.assertEqual(classify_predictor_path([0, 1, 2, 3], 4), "standard")
		self.assertEqual(classify_predictor_path([0, 2, 3], 4), "skip_only")
		self.assertEqual(classify_predictor_path([0, 1, 1, 2, 3], 4), "repeat_only")
		self.assertEqual(classify_predictor_path([0, 0, 2, 3], 4), "skip_repeat")

	def test_rejects_a_path_the_official_parser_cannot_represent(self) -> None:
		with self.assertRaises(ValueError):
			classify_predictor_path([1, 0, 2, 3], 4)


class SmokeSummaryTest(unittest.TestCase):
	def test_accepts_multiple_trace_shards(self) -> None:
		arguments = parse_args(
			["--trace-jsonl", "shard-0.jsonl", "shard-1.jsonl"],
		)

		self.assertEqual(
			arguments.trace_jsonl,
			[Path("shard-0.jsonl"), Path("shard-1.jsonl")],
		)

	def test_requires_deep_search_and_all_three_valid_program_kinds(self) -> None:
		records = [
			{
				"question_id": "q1",
				"final_valid_transitions": [
					[0, 2, 3],
					[0, 1, 1, 2, 3],
					[0, 0, 2, 3],
				],
				"final_invalid_transitions": [[0, 1, 2, 3]],
				"search_metadata": {
					"mode": "predictor_compatible",
					"maximum_tree_depth_reached": 3,
					"tree_policy_selection_count": 7,
				},
			},
		]

		summary = summarize_trace_records(records, original_depth=4)

		self.assertEqual(summary["evaluated_path_counts"]["skip_repeat"], 1)
		self.assertEqual(summary["valid_path_counts"]["skip_only"], 1)
		self.assertEqual(summary["valid_path_counts"]["repeat_only"], 1)
		self.assertEqual(summary["valid_path_counts"]["skip_repeat"], 1)
		self.assertEqual(smoke_failures(summary), ())

	def test_reports_missing_joint_valid_programs(self) -> None:
		records = [
			{
				"question_id": "q1",
				"final_valid_transitions": [[0, 2, 3], [0, 1, 1, 2, 3]],
				"final_invalid_transitions": [[0, 0, 2, 3]],
				"search_metadata": {
					"mode": "predictor_compatible",
					"maximum_tree_depth_reached": 2,
					"tree_policy_selection_count": 2,
				},
			},
		]

		summary = summarize_trace_records(records, original_depth=4)

		self.assertIn(
			"no valid skip_repeat path was found",
			smoke_failures(summary),
		)


if __name__ == "__main__":
	unittest.main()
