import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_predictor_eval_2gpu import _summarize_pass_at_k, build_evaluation_command


class TwoGpuPredictorEvaluationCommandTest(unittest.TestCase):
	def test_uses_recorded_test_split_and_forces_online_path_execution(self) -> None:
		command = build_evaluation_command(
			python_executable=Path("/repo/.venv/bin/python"),
			repository_root=Path("/repo"),
			predictor_data_root=Path("/predictor-data"),
			output_root=Path("/eval"),
			hf_cache_dir=Path("/hf"),
			checkpoint_path=Path("/checkpoints/diff-3.pt"),
			difficulty=3,
		)

		joined = " ".join(command)
		self.assertIn("--eval --policy_mode polar --target_diff 3", joined)
		self.assertIn("--use_recorded_data_splits", command)
		self.assertIn("--no_trust_valid_cache", command)
		self.assertIn("--num_samples 375", joined)
		self.assertIn("--beam_size 5", joined)
		self.assertIn("--top_k_paths 5", joined)
		self.assertIn("--max_new_tokens 50", joined)
		self.assertIn("--checkpoint_path /checkpoints/diff-3.pt", joined)

	def test_summarizes_prefix_pass_at_k_without_more_model_calls(self) -> None:
		with TemporaryDirectory() as directory:
			root = Path(directory)
			rows = [
				{"top_paths": [{"score": 0.0}, {"score": 1.0}]},
				{"top_paths": [{"score": 1.0}, {"score": 0.0}]},
			]
			for difficulty in range(1, 6):
				result_path = root / f"diff-{difficulty}" / "eval_results_test.json"
				result_path.parent.mkdir(parents=True)
				result_path.write_text(__import__("json").dumps(rows), encoding="utf-8")

			summary = _summarize_pass_at_k(root)

		self.assertEqual(summary["1"]["pass_at_k"]["1"], 0.5)
		self.assertEqual(summary["1"]["pass_at_k"]["2"], 1.0)


if __name__ == "__main__":
	unittest.main()
