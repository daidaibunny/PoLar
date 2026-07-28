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


def fake_scores(score_inputs):
	return tuple(fake_score(*score_input) for score_input in score_inputs)


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
				score_generations=fake_scores,
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
				"score_generations": fake_scores,
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

	def test_reuses_scores_for_identical_question_answer_pairs(self) -> None:
		calls = []

		def repeated_answer_scores(score_inputs):
			calls.extend(score_inputs)
			return tuple(
				EvaluationResult(
					binary_reward=1,
					generated_answer=generated_answer,
				)
				for _, _, generated_answer in score_inputs
			)

		class RepeatedAnswerExecutor(FakeBatchExecutor):
			def generate_batch(self, questions, path, settings):
				self.calls.append((tuple(questions), tuple(path), settings))
				return BatchedGeneration(
					prompts=tuple(f"prompt:{question}" for question in questions),
					path=tuple(path),
					generated_answers=tuple("\\boxed{1}" for _ in questions),
				)

		with tempfile.TemporaryDirectory() as directory:
			generate_mcts_records(
				samples=self.samples,
				executor=RepeatedAnswerExecutor(),
				cache=JsonlEvaluationCache(Path(directory) / "cache.jsonl"),
				search_config=self.search_config,
				search_mode=SearchMode.PREDICTOR_COMPATIBLE,
				model_id="model",
				model_revision="model-revision",
				tokenizer_revision="tokenizer-revision",
				batch_size=2,
				score_generations=repeated_answer_scores,
				original_depth=4,
			)

		self.assertEqual(len(calls), 2)

	def test_batches_reward_evaluation_across_diverged_paths_in_one_round(self) -> None:
		samples = tuple(
			SearchSample(
				question_id=f"q{index}",
				question=f"question-{index}",
				ground_truth=str((index % 4) + 1),
			)
			for index in range(8)
		)
		score_batches = []

		def recording_scores(score_inputs):
			score_batches.append(tuple(score_inputs))
			return fake_scores(score_inputs)

		with tempfile.TemporaryDirectory() as directory:
			executor = FakeBatchExecutor()
			result = generate_mcts_records(
				samples=samples,
				executor=executor,
				cache=JsonlEvaluationCache(Path(directory) / "cache.jsonl"),
				search_config=MCTSConfig(n_simulations=20, seed=42),
				search_mode=SearchMode.PREDICTOR_COMPATIBLE,
				model_id="model",
				model_revision="model-revision",
				tokenizer_revision="tokenizer-revision",
				batch_size=len(samples),
				score_generations=recording_scores,
				original_depth=4,
			)

		self.assertTrue(any(len(call[0]) < len(samples) for call in executor.calls))
		self.assertTrue(score_batches)
		self.assertTrue(all(len(batch) == len(samples) for batch in score_batches))
		self.assertEqual(result.reward_evaluator_batches, len(score_batches))
		self.assertLess(result.reward_evaluator_batches, result.model_batches)

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
				"score_generations": fake_scores,
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
