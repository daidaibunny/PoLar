import math
import unittest

from reconstruction.mcts import (
    Action,
    EvaluationResult,
    MCTSConfig,
    ProgramMCTS,
    SearchMode,
    apply_action,
    enumerate_actions,
    is_predictor_compatible_path,
    ucb_score,
)


class ActionTest(unittest.TestCase):
    def test_skip_removes_a_contiguous_block(self) -> None:
        action = Action(operation="skip", start=1, block_length=2)

        self.assertEqual(apply_action((0, 1, 2, 3), action), (0, 3))

    def test_repeat_adds_the_requested_number_of_extra_copies(self) -> None:
        action = Action(
            operation="repeat",
            start=1,
            block_length=2,
            repeat_count=2,
        )

        self.assertEqual(
            apply_action((0, 1, 2, 3), action),
            (0, 1, 2, 1, 2, 1, 2, 3),
        )

    def test_diagnostic_actions_respect_block_and_repeat_limits(self) -> None:
        actions = enumerate_actions(
            (0, 1, 2, 3),
            mode=SearchMode.DIAGNOSTIC,
            max_block_length=2,
            max_repeat_count=2,
            original_depth=4,
        )

        self.assertTrue(actions)
        self.assertLessEqual(max(action.block_length for action in actions), 2)
        self.assertLessEqual(max(action.repeat_count for action in actions), 2)


class PredictorCompatibilityTest(unittest.TestCase):
    def test_accepts_skip_keep_and_one_extra_segment_execution(self) -> None:
        self.assertTrue(is_predictor_compatible_path((0, 1, 1, 2, 3), 4))
        self.assertTrue(is_predictor_compatible_path((0, 1, 0, 1, 2, 3), 4))
        self.assertTrue(is_predictor_compatible_path((0, 2, 3), 4))

    def test_rejects_deep_repeat_and_reordering(self) -> None:
        self.assertFalse(is_predictor_compatible_path((0, 1, 1, 1, 2, 3), 4))
        self.assertFalse(is_predictor_compatible_path((0, 2, 1, 3), 4))


class TreePolicyTest(unittest.TestCase):
    def test_ucb_keeps_binary_reward_separate_from_length_penalty(self) -> None:
        score = ucb_score(
            reward_sum=3.0,
            visits=4,
            parent_visits=10,
            path_length=2,
            original_depth=4,
            exploration_constant=math.sqrt(2),
            length_penalty_lambda=5.0,
        )
        expected = 0.75 + math.sqrt(2) * math.sqrt(math.log(10) / 4) - 2.5

        self.assertAlmostEqual(score, expected)

    def test_search_evaluates_each_question_path_at_most_once(self) -> None:
        calls: list[tuple[int, ...]] = []

        def evaluator(path: tuple[int, ...]) -> EvaluationResult:
            calls.append(path)
            return EvaluationResult(
                binary_reward=int(2 not in path),
                generated_answer="\\boxed{1}" if 2 not in path else "\\boxed{0}",
            )

        config = MCTSConfig(
            n_simulations=20,
            max_block_length=2,
            max_repeat_count_diagnostic=2,
            max_repeat_count_predictor=1,
            seed=42,
        )
        result = ProgramMCTS(
            original_depth=4,
            config=config,
            mode=SearchMode.DIAGNOSTIC,
        ).search(evaluator)

        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(result.initial_score, 0)
        self.assertTrue(result.valid_programs)
        self.assertTrue(any(2 not in path for path in result.valid_programs))
        self.assertEqual(result.search_metadata["n_simulations"], 20)


if __name__ == "__main__":
    unittest.main()
