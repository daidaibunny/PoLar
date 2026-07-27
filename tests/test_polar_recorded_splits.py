import json
import tempfile
import unittest
from pathlib import Path

from polar.data import PolarDataset
from run_polar import build_arg_parser


class RecordedDataSplitTest(unittest.TestCase):
	def test_cli_exposes_recorded_splits_and_train_only_mode(self) -> None:
		arguments = build_arg_parser().parse_args(
			["--use_recorded_data_splits", "--train_only"],
		)

		self.assertTrue(arguments.use_recorded_data_splits)
		self.assertTrue(arguments.train_only)

	def test_filters_public_records_by_explicit_search_metadata_split(self) -> None:
		def sample(question: str, split: str):
			return {
				"question": question,
				"gt_ans": "1",
				"final_valid_transitions": [[0, 1, 2, 3]],
				"search_metadata": {"data_split": split},
			}

		payload = {
			"samples": [
				sample("train-one", "train"),
				sample("validation-one", "validation"),
				sample("train-two", "train"),
			],
		}
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "merged.json"
			path.write_text(json.dumps(payload), encoding="utf-8")

			train = PolarDataset(
				merged_samples_json=str(path),
				start_idx=0,
				end_idx=3,
				original_depth=4,
				split_filter="train",
			)
			validation = PolarDataset(
				merged_samples_json=str(path),
				start_idx=0,
				end_idx=3,
				original_depth=4,
				split_filter="validation",
			)

		self.assertEqual([row["question"] for row in train.examples], ["train-one", "train-two"])
		self.assertEqual([row["question"] for row in validation.examples], ["validation-one"])


if __name__ == "__main__":
	unittest.main()
