import unittest

import torch

from reconstruction.executor_validation import compare_full_path_results


class FullPathComparisonTest(unittest.TestCase):
	def test_accepts_identical_tokens_close_logits_and_exact_layer_order(self) -> None:
		result = compare_full_path_results(
			baseline_sequences=torch.tensor([[1, 2, 3]]),
			patched_sequences=torch.tensor([[1, 2, 3]]),
			baseline_logits=torch.tensor([[1.0, 2.0]]),
			patched_logits=torch.tensor([[1.0, 2.000001]]),
			baseline_layer_order=(0, 1, 2),
			patched_layer_order=(0, 1, 2),
			expected_path=(0, 1, 2),
			torch_module=torch,
		)

		self.assertTrue(result.passed)
		self.assertTrue(result.generated_tokens_identical)
		self.assertTrue(result.first_token_logits_close)
		self.assertAlmostEqual(result.maximum_logit_absolute_difference, 0.000001)

	def test_rejects_any_token_or_layer_order_difference(self) -> None:
		result = compare_full_path_results(
			baseline_sequences=torch.tensor([[1, 2, 3]]),
			patched_sequences=torch.tensor([[1, 2, 4]]),
			baseline_logits=torch.tensor([[1.0, 2.0]]),
			patched_logits=torch.tensor([[1.0, 2.0]]),
			baseline_layer_order=(0, 1, 2),
			patched_layer_order=(0, 2, 1),
			expected_path=(0, 1, 2),
			torch_module=torch,
		)

		self.assertFalse(result.passed)
		self.assertFalse(result.generated_tokens_identical)
		self.assertFalse(result.patched_layer_order_correct)


if __name__ == "__main__":
	unittest.main()
