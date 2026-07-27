import tempfile
import unittest
from pathlib import Path

from reconstruction.cache import (
    CacheIdentity,
    CachedEvaluation,
    JsonlEvaluationCache,
    make_cache_key,
)


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


if __name__ == "__main__":
    unittest.main()
