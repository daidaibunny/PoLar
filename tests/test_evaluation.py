import unittest
from unittest.mock import patch

from reconstruction.evaluation import (
	FULL_PATH,
	GenerationSettings,
	SAMPLING_TEMPERATURES,
	build_direct_prompt,
	prompt_sha256,
	sampling_settings,
	score_math_generation,
	score_math_generations,
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

	def test_sampling_uses_only_the_reported_temperature_grid(self) -> None:
		self.assertEqual(SAMPLING_TEMPERATURES, (0.3, 0.7, 1.0))
		settings = sampling_settings(temperature=0.7, pass_k=5)

		self.assertTrue(settings.do_sample)
		self.assertEqual(settings.temperature, 0.7)
		self.assertEqual(settings.num_return_sequences, 5)
		with self.assertRaises(ValueError):
			sampling_settings(temperature=0.5, pass_k=5)


class OfficialMathEvaluatorTest(unittest.TestCase):
	@patch("dart_math.eval.EvaluatorMathBatch")
	def test_batch_scoring_matches_official_polar_configuration(
		self,
		evaluator_class,
	) -> None:
		evaluator = evaluator_class.return_value
		evaluator.batch_eval.return_value = (["2", "3"], [True, False])
		inputs = (
			("ignored question one", "2", "\\boxed{2}"),
			("ignored question two", "2", "\\boxed{3}"),
		)

		results = score_math_generations(inputs)

		evaluator_class.assert_called_once_with(
			strict_extract=True,
			use_orig_eq_for_olympiadbench=True,
			timeout=60,
		)
		samples = evaluator.batch_eval.call_args.args[0]
		self.assertEqual(evaluator.batch_eval.call_args.kwargs, {"n_procs": 4})
		self.assertEqual([sample.resp for sample in samples], ["\\boxed{2}", "\\boxed{3}"])
		self.assertEqual([sample.ref_ans for sample in samples], ["2", "2"])
		self.assertEqual([sample.query for sample in samples], ["", ""])
		self.assertEqual([sample.dataset for sample in samples], ["math", "math"])
		self.assertEqual([result.binary_reward for result in results], [1, 0])

	def test_accepts_mathematically_equivalent_boxed_answers(self) -> None:
		result = score_math_generation(
			question="Express one half as a decimal.",
			ground_truth="0.5",
			generated_answer="\\boxed{\\frac{1}{2}}",
		)

		self.assertEqual(result.binary_reward, 1)

	def test_matches_official_non_boxed_and_incorrect_answer_behavior(self) -> None:
		self.assertEqual(
			score_math_generation("Question", "2", "2").binary_reward,
			1,
		)
		self.assertEqual(
			score_math_generation("Question", "2", "\\boxed{3}").binary_reward,
			0,
		)

	def test_treats_unparseable_boxed_generation_as_incorrect(self) -> None:
		generated_answer = "\\boxed{2. Thereasonisthatthere}"

		result = score_math_generation(
			question="Question",
			ground_truth="2",
			generated_answer=generated_answer,
		)

		self.assertEqual(result.binary_reward, 0)
		self.assertEqual(result.generated_answer, generated_answer)


if __name__ == "__main__":
	unittest.main()
