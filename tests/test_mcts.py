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
        self.assertFalse(is_predictor_compatible_path((), 4))


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
        self.assertEqual(result.search_metadata["ucb_c"], math.sqrt(2))
        self.assertEqual(result.search_metadata["random_exploration"], 0.1)
        self.assertEqual(
            result.search_metadata["ucb_parent_visit_definition"],
            "immediate parent node visits",
        )

    def test_predictor_repeat_limit_cannot_drift_from_official_parser(self) -> None:
        with self.assertRaises(ValueError):
            MCTSConfig(max_repeat_count_predictor=2)

    def test_stepwise_session_matches_callback_search(self) -> None:
        def evaluator(path: tuple[int, ...]) -> EvaluationResult:
            return EvaluationResult(
                binary_reward=int(len(path) <= 3),
                generated_answer=f"path-length={len(path)}",
            )

        search = ProgramMCTS(
            original_depth=4,
            config=MCTSConfig(n_simulations=20, seed=7),
            mode=SearchMode.PREDICTOR_COMPATIBLE,
        )
        direct_result = search.search(evaluator)
        session = search.start_session()
        while True:
            path = session.request_path()
            if path is None:
                break
            session.record_evaluation(evaluator(path))

        self.assertEqual(session.result(), direct_result)

    def test_stepwise_session_requires_each_requested_path_to_be_recorded(self) -> None:
        session = ProgramMCTS(
            original_depth=4,
            config=MCTSConfig(n_simulations=1),
            mode=SearchMode.PREDICTOR_COMPATIBLE,
        ).start_session()

        self.assertEqual(session.request_path(), (0, 1, 2, 3))
        with self.assertRaises(RuntimeError):
            session.request_path()

    def test_existing_child_is_selected_before_more_root_expansion_when_probability_zero(
        self,
    ) -> None:
        result = ProgramMCTS(
            original_depth=4,
            config=MCTSConfig(
                n_simulations=2,
                random_action_probability=0.0,
                seed=42,
            ),
            mode=SearchMode.PREDICTOR_COMPATIBLE,
        ).search(
            lambda path: EvaluationResult(
                binary_reward=0,
                generated_answer="\\boxed{0}",
            ),
        )

        self.assertEqual(result.search_metadata["maximum_tree_depth_reached"], 2)
        self.assertGreater(result.search_metadata["tree_policy_selection_count"], 0)

    def test_200_predictor_simulations_reach_joint_skip_repeat_paths(self) -> None:
        result = ProgramMCTS(
            original_depth=28,
            config=MCTSConfig(n_simulations=200, seed=42),
            mode=SearchMode.PREDICTOR_COMPATIBLE,
        ).search(
            lambda path: EvaluationResult(
                binary_reward=0,
                generated_answer="\\boxed{0}",
            ),
        )

        self.assertGreater(result.search_metadata["root_action_count"], 200)
        self.assertGreater(result.search_metadata["maximum_tree_depth_reached"], 1)
        self.assertGreater(result.search_metadata["tree_policy_selection_count"], 0)

        path_kinds = set()
        expected_layers = set(range(28))
        for path, _ in result.evaluations:
            has_skip = set(path) != expected_layers
            has_repeat = len(path) != len(set(path))
            if has_skip and has_repeat:
                path_kinds.add("skip_repeat")
            elif has_skip:
                path_kinds.add("skip_only")
            elif has_repeat:
                path_kinds.add("repeat_only")

        self.assertEqual(path_kinds, {"skip_only", "repeat_only", "skip_repeat"})


if __name__ == "__main__":
    unittest.main()
