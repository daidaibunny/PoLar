import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reconstruction.mcts import EvaluationResult, MCTSConfig, ProgramMCTS, SearchMode
from reconstruction.supervision import (
	predictor_supervision_record,
	search_result_record,
	validate_with_official_parser,
	write_merged_mcts_samples,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SupervisionTest(unittest.TestCase):
	def setUp(self) -> None:
		def evaluator(path: tuple[int, ...]) -> EvaluationResult:
			return EvaluationResult(
				binary_reward=int(path in ((0, 1, 2, 3), (0, 1, 1, 2, 3))),
				generated_answer="\\boxed{1}",
			)

		result = ProgramMCTS(
			original_depth=4,
			config=MCTSConfig(n_simulations=30, seed=42),
			mode=SearchMode.PREDICTOR_COMPATIBLE,
		).search(evaluator)
		self.record = search_result_record(
			question_id="q1",
			question="Question",
			ground_truth="1",
			result=result,
			additional_metadata={"model_revision": "revision"},
		)

	def test_search_record_contains_only_actually_executed_programs(self) -> None:
		valid_paths = self.record["final_valid_transitions"]

		self.assertIn([0, 1, 2, 3], valid_paths)
		self.assertEqual(len(valid_paths), len({tuple(path) for path in valid_paths}))

	def test_predictor_record_filters_incompatible_and_duplicate_paths(self) -> None:
		self.record["final_valid_transitions"].extend(
			[
				[0, 1, 1, 1, 2, 3],
				[0, 2, 1, 3],
				[0, 1, 2, 3],
			],
		)
		predictor_record = predictor_supervision_record(self.record, original_depth=4)

		paths = predictor_record["final_valid_transitions"]
		self.assertEqual(len(paths), len({tuple(path) for path in paths}))
		self.assertNotIn([0, 1, 1, 1, 2, 3], paths)
		self.assertNotIn([0, 2, 1, 3], paths)

	def test_writer_uses_the_official_validation_gate(self) -> None:
		predictor_record = predictor_supervision_record(self.record, original_depth=4)

		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "merged_mcts_samples.json"
			with patch(
				"reconstruction.supervision.validate_with_official_parser",
			) as validate:
				write_merged_mcts_samples(path, [predictor_record], original_depth=4)

			self.assertTrue(path.is_file())
			validate.assert_called_once_with([predictor_record], 4)

	@unittest.skipUnless(TORCH_AVAILABLE, "official parser requires the local torch package")
	def test_official_parser_accepts_predictor_paths(self) -> None:
		predictor_record = predictor_supervision_record(self.record, original_depth=4)

		validate_with_official_parser([predictor_record], original_depth=4)


if __name__ == "__main__":
	unittest.main()
