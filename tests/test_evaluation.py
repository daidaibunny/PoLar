import unittest

from reconstruction.evaluation import (
	FULL_PATH,
	GenerationSettings,
	build_direct_prompt,
	prompt_sha256,
	score_math_generation,
)


class DirectPromptTest(unittest.TestCase):
	def test_prompt_matches_the_frozen_direct_answer_format(self) -> None:
		question = "What is 1+1?"
		expected = (
			"Solve the following math problem and output ONLY the final answer directly, "
			"formatted strictly as \\boxed{ANSWER}.\n"
			"### Problem Start\n"
			"What is 1+1?\n"
			"### Problem End\n"
			"Answer:"
		)

		self.assertEqual(build_direct_prompt(question), expected)
		self.assertEqual(FULL_PATH, tuple(range(28)))
		self.assertEqual(len(prompt_sha256(question)), 64)

	def test_greedy_generation_settings_are_explicit_and_hashable(self) -> None:
		settings = GenerationSettings()

		self.assertEqual(
			settings.generate_kwargs(),
			{
				"max_new_tokens": 50,
				"do_sample": False,
				"num_return_sequences": 1,
			},
		)
		self.assertEqual(settings.sha256(), GenerationSettings().sha256())


class OfficialMathEvaluatorTest(unittest.TestCase):
	def test_accepts_mathematically_equivalent_boxed_answers(self) -> None:
		result = score_math_generation(
			question="Express one half as a decimal.",
			ground_truth="0.5",
			generated_answer="\\boxed{\\frac{1}{2}}",
		)

		self.assertEqual(result.binary_reward, 1)

	def test_rejects_non_boxed_or_incorrect_answers(self) -> None:
		self.assertEqual(
			score_math_generation("Question", "2", "2").binary_reward,
			0,
		)
		self.assertEqual(
			score_math_generation("Question", "2", "\\boxed{3}").binary_reward,
			0,
		)


if __name__ == "__main__":
	unittest.main()

