import tempfile
import unittest
from pathlib import Path

from reconstruction.cache import (
    CacheIdentity,
    CachedEvaluation,
    CachedPathEvaluator,
    JsonlEvaluationCache,
    make_cache_key,
)
from reconstruction.mcts import EvaluationResult


class CacheKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = CacheIdentity(
            question_id="q1",
            path=(0, 1, 1, 2),
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            model_revision="model-revision",
            tokenizer_revision="tokenizer-revision",
            prompt_hash="prompt-hash",
            generation_config_hash="generation-hash",
        )

    def test_cache_key_is_stable(self) -> None:
        self.assertEqual(make_cache_key(self.identity), make_cache_key(self.identity))

    def test_cache_key_changes_with_path(self) -> None:
        changed = CacheIdentity(**{**self.identity.__dict__, "path": (0, 1, 2)})

        self.assertNotEqual(make_cache_key(self.identity), make_cache_key(changed))

    def test_jsonl_cache_reloads_without_re_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            cache = JsonlEvaluationCache(path)
            evaluation = CachedEvaluation(
                identity=self.identity,
                binary_reward=1,
                generated_answer="\\boxed{42}",
            )
            cache.put(evaluation)

            reloaded = JsonlEvaluationCache(path)

            self.assertEqual(reloaded.get(self.identity), evaluation)
            self.assertFalse(reloaded.put(evaluation))

    def test_cache_can_close_and_reopen_its_append_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            cache = JsonlEvaluationCache(path)
            first = CachedEvaluation(
                identity=self.identity,
                binary_reward=1,
                generated_answer="\\boxed{42}",
            )
            second_identity = CacheIdentity(
                **{**self.identity.__dict__, "path": (0, 1, 2)},
            )
            second = CachedEvaluation(
                identity=second_identity,
                binary_reward=0,
                generated_answer="\\boxed{0}",
            )

            cache.put(first)
            cache.close()
            cache.put(second)
            cache.close()

            reloaded = JsonlEvaluationCache(path)
            self.assertEqual(reloaded.get(self.identity), first)
            self.assertEqual(reloaded.get(second_identity), second)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_cached_path_evaluator_replays_without_calling_executor(self) -> None:
        calls = []

        def execute(path: tuple[int, ...]) -> EvaluationResult:
            calls.append(path)
            return EvaluationResult(binary_reward=1, generated_answer="\\boxed{42}")

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.jsonl"
            arguments = {
                "question_id": "q1",
                "model_id": "model",
                "model_revision": "model-revision",
                "tokenizer_revision": "tokenizer-revision",
                "prompt_hash": "prompt-hash",
                "generation_config_hash": "generation-hash",
                "execute": execute,
            }
            first = CachedPathEvaluator(JsonlEvaluationCache(cache_path), **arguments)

            self.assertEqual(first((0, 1)).binary_reward, 1)
            second = CachedPathEvaluator(JsonlEvaluationCache(cache_path), **arguments)
            self.assertEqual(second((0, 1)).binary_reward, 1)
            self.assertEqual(calls, [(0, 1)])
            self.assertEqual(second.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
