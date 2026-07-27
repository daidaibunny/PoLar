import json
import tempfile
import unittest
from pathlib import Path

from reconstruction.cache import CacheIdentity, CachedEvaluation, JsonlEvaluationCache
from reconstruction.mcts import EvaluationResult
from scripts.rescore_mcts_cache import rescore_cache


class RescoreMctsCacheTest(unittest.TestCase):
	def test_preserves_answers_and_identity_while_backing_up_old_rewards(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			data_path = root / "diff-1.json"
			data_path.write_text(
				json.dumps(
					{
						"train": [
							{"query_id": "q1", "question": "one", "gt_ans": "1"},
							{"query_id": "q2", "question": "two", "gt_ans": "2"},
						],
						"validation": [],
					},
				),
				encoding="utf-8",
			)
			cache_path = root / "cache.jsonl"
			cache = JsonlEvaluationCache(cache_path)
			for index, question_id in enumerate(("q1", "q2"), start=1):
				cache.put(
					CachedEvaluation(
						identity=CacheIdentity(
							question_id=question_id,
							path=(0, 1),
							model_id="model",
							model_revision="model-revision",
							tokenizer_revision="tokenizer-revision",
							prompt_hash=f"prompt-{index}",
							generation_config_hash="generation",
						),
						binary_reward=0,
						generated_answer=f"answer-{index}",
					),
				)
			cache.close()
			original_bytes = cache_path.read_bytes()

			def fake_scores(score_inputs):
				return tuple(
					EvaluationResult(
						binary_reward=1,
						generated_answer=generated_answer,
					)
					for _, _, generated_answer in score_inputs
				)

			summary = rescore_cache(
				cache_path=cache_path,
				data_file=data_path,
				batch_size=2,
				backup_suffix="old",
				score_generations=fake_scores,
			)

			backup_path = root / "cache.jsonl.pre-official-old.bak"
			self.assertEqual(backup_path.read_bytes(), original_bytes)
			payloads = [
				json.loads(line)
				for line in cache_path.read_text(encoding="utf-8").splitlines()
			]
			self.assertEqual([payload["binary_reward"] for payload in payloads], [1, 1])
			self.assertEqual(
				[payload["generated_answer"] for payload in payloads],
				["answer-1", "answer-2"],
			)
			self.assertEqual(summary["entries"], 2)
			self.assertEqual(summary["changed_rewards"], 2)


if __name__ == "__main__":
	unittest.main()
