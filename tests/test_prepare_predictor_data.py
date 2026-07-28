import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_predictor_data import prepare_difficulty


class PreparePredictorDataTest(unittest.TestCase):
	def test_combines_supervised_train_validation_with_unsearched_test(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			supervision_path = root / "supervision.json"
			data_path = root / "diff-1.json"
			output_path = root / "output.json"
			supervision_path.write_text(
				json.dumps(
					{
						"samples": [
							self._supervised_record("train-1", "train"),
							self._supervised_record("validation-1", "validation"),
						],
					},
				),
				encoding="utf-8",
			)
			data_path.write_text(
				json.dumps(
					{
						"train": [self._source_record("train-1")],
						"validation": [self._source_record("validation-1")],
						"test": [self._source_record("test-1")],
					},
				),
				encoding="utf-8",
			)

			summary = prepare_difficulty(
				supervision_path=supervision_path,
				data_path=data_path,
				output_path=output_path,
				original_depth=4,
				expected_split_counts={"train": 1, "validation": 1, "test": 1},
			)

			output = json.loads(output_path.read_text(encoding="utf-8"))["samples"]
			self.assertEqual([row["question_id"] for row in output], [
				"train-1",
				"validation-1",
				"test-1",
			])
			test_record = output[-1]
			self.assertEqual(test_record["final_valid_transitions"], [])
			self.assertEqual(test_record["final_invalid_transitions"], [])
			self.assertFalse(test_record["search_metadata"]["mcts_searched"])
			self.assertEqual(summary["split_counts"]["test"], 1)

	@staticmethod
	def _source_record(question_id: str) -> dict:
		return {
			"query_id": question_id,
			"question": f"question-{question_id}",
			"gt_ans": "1",
		}

	@classmethod
	def _supervised_record(cls, question_id: str, split: str) -> dict:
		record = cls._source_record(question_id)
		return {
			"question_id": question_id,
			"question": record["question"],
			"gt_ans": record["gt_ans"],
			"initial_score": 0,
			"final_valid_transitions": [[0, 1, 2, 3]],
			"search_metadata": {"data_split": split},
		}


if __name__ == "__main__":
	unittest.main()
