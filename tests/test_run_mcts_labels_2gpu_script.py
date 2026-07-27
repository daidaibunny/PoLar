import unittest
from pathlib import Path

from scripts.run_mcts_labels_2gpu import build_label_command


class TwoGpuLabelCommandTest(unittest.TestCase):
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


if __name__ == "__main__":
	unittest.main()
