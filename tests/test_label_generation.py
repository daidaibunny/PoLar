import json
import tempfile
import unittest
from pathlib import Path

from reconstruction.cache import JsonlEvaluationCache
from reconstruction.executor import BatchedGeneration
from reconstruction.label_generation import (
	SearchSample,
	append_records_jsonl,
	generate_mcts_records,
	load_search_samples,
	read_completed_question_ids,
)
from reconstruction.mcts import EvaluationResult, MCTSConfig, SearchMode


class FakeBatchExecutor:
	def __init__(self) -> None:
		self.calls = []

	def generate_batch(self, questions, path, settings):
		self.calls.append((tuple(questions), tuple(path), settings))
		return BatchedGeneration(
			prompts=tuple(f"prompt:{question}" for question in questions),
			path=tuple(path),
			generated_answers=tuple(
				f"\\boxed{{{len(path)}}}" for _ in questions
			),
		)


def fake_score(question, ground_truth, generated_answer):
	return EvaluationResult(
		binary_reward=int(generated_answer == f"\\boxed{{{ground_truth}}}"),
		generated_answer=generated_answer,
	)


class BatchedLabelGenerationTest(unittest.TestCase):
	def setUp(self) -> None:
		self.samples = (
			SearchSample(question_id="q1", question="one", ground_truth="4"),
			SearchSample(question_id="q2", question="two", ground_truth="3"),
		)
		self.search_config = MCTSConfig(n_simulations=2, seed=42)

	def test_groups_identical_paths_across_questions(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			executor = FakeBatchExecutor()
			result = generate_mcts_records(
				samples=self.samples,
				executor=executor,
				cache=JsonlEvaluationCache(Path(directory) / "cache.jsonl"),
				search_config=self.search_config,
				search_mode=SearchMode.PREDICTOR_COMPATIBLE,
				model_id="model",
				model_revision="model-revision",
				tokenizer_revision="tokenizer-revision",
				batch_size=2,
				score_generation=fake_score,
				original_depth=4,
			)

		self.assertEqual(len(result.records), 2)
		self.assertEqual(len(executor.calls), 3)
		self.assertTrue(all(len(call[0]) == 2 for call in executor.calls))
		self.assertEqual(result.cache_misses, 6)
		self.assertEqual(result.cache_hits, 0)
		self.assertEqual(result.records[0]["question_id"], "q1")
		self.assertEqual(result.records[0]["initial_score"], 1)
		self.assertEqual(result.records[1]["initial_score"], 0)

	def test_replays_every_question_path_without_model_execution(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			cache_path = Path(directory) / "cache.jsonl"
			arguments = {
				"samples": self.samples,
				"search_config": self.search_config,
				"search_mode": SearchMode.PREDICTOR_COMPATIBLE,
				"model_id": "model",
				"model_revision": "model-revision",
				"tokenizer_revision": "tokenizer-revision",
				"batch_size": 2,
				"score_generation": fake_score,
				"original_depth": 4,
			}
			first_executor = FakeBatchExecutor()
			first = generate_mcts_records(
				executor=first_executor,
				cache=JsonlEvaluationCache(cache_path),
				**arguments,
			)
			second_executor = FakeBatchExecutor()
			second = generate_mcts_records(
				executor=second_executor,
				cache=JsonlEvaluationCache(cache_path),
				**arguments,
			)

		self.assertEqual(second.records, first.records)
		self.assertEqual(second_executor.calls, [])
		self.assertEqual(second.cache_hits, 6)
		self.assertEqual(second.cache_misses, 0)

	def test_rejects_invalid_batch_and_duplicate_question_ids(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			arguments = {
				"samples": self.samples,
				"executor": FakeBatchExecutor(),
				"cache": JsonlEvaluationCache(Path(directory) / "cache.jsonl"),
				"search_config": self.search_config,
				"search_mode": SearchMode.PREDICTOR_COMPATIBLE,
				"model_id": "model",
				"model_revision": "model-revision",
				"tokenizer_revision": "tokenizer-revision",
				"score_generation": fake_score,
				"original_depth": 4,
			}
			with self.assertRaises(ValueError):
				generate_mcts_records(batch_size=0, **arguments)
			with self.assertRaises(ValueError):
				generate_mcts_records(
					batch_size=1,
					**{
						**arguments,
						"samples": (self.samples[0], self.samples[0]),
					},
				)


class LabelInputOutputTest(unittest.TestCase):
	def test_loads_requested_splits_then_applies_global_limit_and_shard(self) -> None:
		payload = {
			"difficulty_level": 1,
			"train": [
				{"query_id": "q1", "question": "one", "gt_ans": "1"},
				{"query_id": "q2", "question": "two", "gt_ans": "2"},
			],
			"validation": [
				{"query_id": "q3", "question": "three", "gt_ans": "3"},
			],
			"test": [
				{"query_id": "q4", "question": "four", "gt_ans": "4"},
			],
		}
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "diff-1.json"
			path.write_text(json.dumps(payload), encoding="utf-8")

			samples = load_search_samples(
				path=path,
				splits=("train", "validation"),
				max_samples=3,
				num_shards=2,
				shard_index=1,
			)

		self.assertEqual([sample.question_id for sample in samples], ["q2"])
		self.assertEqual(samples[0].additional_metadata["data_split"], "train")
		self.assertEqual(samples[0].additional_metadata["difficulty_level"], 1)

	def test_appends_records_and_refuses_duplicate_question_ids(self) -> None:
		records = (
			{"question_id": "q1", "initial_score": 1},
			{"question_id": "q2", "initial_score": 0},
		)
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "records.jsonl"
			append_records_jsonl(path, records)

			self.assertEqual(read_completed_question_ids(path), {"q1", "q2"})
			with self.assertRaises(ValueError):
				append_records_jsonl(path, ({"question_id": "q2"},))
			self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
	unittest.main()
