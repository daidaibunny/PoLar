import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_predictor_training_2gpu import (
	TRAINING_ASSIGNMENTS,
	build_training_command,
	resolve_cached_embedding_revision,
)


class TwoGpuPredictorTrainingCommandTest(unittest.TestCase):
	def test_balances_five_independent_predictors_over_two_gpus(self) -> None:
		self.assertEqual(TRAINING_ASSIGNMENTS, {0: (1, 2), 1: (3, 4, 5)})

	def test_resolves_pinned_embedding_snapshot_from_shared_cache(self) -> None:
		with TemporaryDirectory() as directory:
			hf_home = Path(directory)
			model_root = hf_home / "hub" / "models--Qwen--Qwen3-Embedding-0.6B"
			(model_root / "refs").mkdir(parents=True)
			(model_root / "refs" / "main").write_text("a" * 40, encoding="utf-8")
			(model_root / "snapshots" / ("a" * 40)).mkdir(parents=True)

			revision = resolve_cached_embedding_revision(hf_home)

		self.assertEqual(revision, "a" * 40)

	def test_uses_official_readme_starting_configuration_and_public_splits(self) -> None:
		command = build_training_command(
			python_executable=Path(".venv/bin/python"),
			repository_root=Path("/repo"),
			supervision_root=Path("/supervision"),
			output_root=Path("/outputs"),
			hf_cache_dir=Path("/hf"),
			difficulty=4,
		)

		joined = " ".join(command)
		self.assertIn("--target_diff 4", joined)
		self.assertIn("--num_epochs 10", joined)
		self.assertIn("--batch_size 128", joined)
		self.assertIn("--learning_rate 0.0005", joined)
		self.assertIn("--max_paths_per_sample 50", joined)
		self.assertIn("--per_sample_weight_normalize", command)
		self.assertIn("--reweight_original_path_if_shorter_valid", command)
		self.assertIn("--original_path_weight 0.30", joined)
		self.assertIn("--beam_size 5", joined)
		self.assertIn("--top_k_paths 5", joined)
		self.assertIn("--lr_scheduler cosine", joined)
		self.assertIn("--warmup_steps 10", joined)
		self.assertIn("--seed 42", joined)
		self.assertIn("--use_recorded_data_splits", command)
		self.assertIn("--train_only", command)


if __name__ == "__main__":
	unittest.main()
