import unittest

from reconstruction.metrics import (
	best_sampling_temperature_curve,
	pass_at_k,
	pass_at_k_curve,
)


class PassAtKTest(unittest.TestCase):
	def test_counts_questions_with_any_correct_prefix_sample(self) -> None:
		correctness = [
			[False, True, False],
			[False, False, False],
			[True, False, False],
		]

		self.assertEqual(pass_at_k(correctness, 1), 1 / 3)
		self.assertEqual(pass_at_k(correctness, 2), 2 / 3)
		self.assertEqual(
			pass_at_k_curve(correctness, maximum_k=3),
			{1: 1 / 3, 2: 2 / 3, 3: 2 / 3},
		)

	def test_rejects_rows_with_too_few_samples(self) -> None:
		with self.assertRaises(ValueError):
			pass_at_k([[True]], 2)

	def test_selects_best_temperature_for_each_k(self) -> None:
		selected = best_sampling_temperature_curve(
			{
				0.3: [[True, False], [False, False]],
				0.7: [[False, True], [False, True]],
				1.0: [[False, False], [False, False]],
			},
			maximum_k=2,
		)

		self.assertEqual(selected[1].temperature, 0.3)
		self.assertEqual(selected[1].accuracy, 0.5)
		self.assertEqual(selected[2].temperature, 0.7)
		self.assertEqual(selected[2].accuracy, 1.0)


if __name__ == "__main__":
	unittest.main()
