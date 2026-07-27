"""Batch frozen-model executions across independent MCTS question searches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from reconstruction.cache import (
	CacheIdentity,
	CachedEvaluation,
	JsonlEvaluationCache,
)
from reconstruction.evaluation import (
	GenerationSettings,
	MathScoreInput,
	prompt_sha256,
	score_math_generations,
)
from reconstruction.mcts import (
	EvaluationResult,
	LayerPath,
	MCTSConfig,
	ProgramMCTS,
	ProgramMCTSSession,
	SearchMode,
)
from reconstruction.supervision import search_result_record


@dataclass(frozen=True)
class SearchSample:
	"""One query-level math problem eligible for offline path search."""

	question_id: str
	question: str
	ground_truth: str
	additional_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedSearchResult:
	"""Completed supervision records and cache/model execution counters."""

	records: Tuple[Dict[str, Any], ...]
	cache_hits: int
	cache_misses: int
	model_batches: int


@dataclass
class _ActiveSearch:
	sample: SearchSample
	session: ProgramMCTSSession
	prompt_hash: str


ScoreGenerations = Callable[
	[Sequence[MathScoreInput]],
	Sequence[EvaluationResult],
]


def load_search_samples(
	path: Path,
	splits: Sequence[str],
	max_samples: Optional[int],
	num_shards: int,
	shard_index: int,
) -> Tuple[SearchSample, ...]:
	"""Load query-level records, limit globally, then select one stable shard."""
	path = Path(path)
	allowed_splits = {"train", "validation", "test"}
	if not splits or any(split not in allowed_splits for split in splits):
		raise ValueError(f"splits must come from {sorted(allowed_splits)}")
	if max_samples is not None and max_samples <= 0:
		raise ValueError("max_samples must be positive when provided")
	if num_shards <= 0:
		raise ValueError("num_shards must be positive")
	if shard_index < 0 or shard_index >= num_shards:
		raise ValueError("shard_index must be within [0, num_shards)")

	data_bytes = path.read_bytes()
	payload = json.loads(data_bytes)
	if not isinstance(payload, dict):
		raise ValueError("difficulty data must be a JSON object")
	difficulty_level = payload.get("difficulty_level")
	difficulty_definition = payload.get("difficulty_definition")
	loaded: List[SearchSample] = []
	for split in splits:
		records = payload.get(split)
		if not isinstance(records, list):
			raise ValueError(f"split {split!r} must be a JSON list")
		for record in records:
			if not isinstance(record, dict):
				raise ValueError(f"split {split!r} contains a non-object record")
			loaded.append(
				SearchSample(
					question_id=str(record.get("query_id", "")),
					question=str(record.get("question", "")),
					ground_truth=str(record.get("gt_ans", "")),
					additional_metadata={
						"data_split": split,
						"difficulty_level": difficulty_level,
						"difficulty_definition": difficulty_definition,
						"data_file": str(path),
						"data_file_sha256": hashlib.sha256(data_bytes).hexdigest(),
					},
				),
			)
	question_ids = [sample.question_id for sample in loaded]
	if len(question_ids) != len(set(question_ids)):
		raise ValueError("selected data splits contain duplicate query_id values")
	limited = loaded[:max_samples] if max_samples is not None else loaded
	return tuple(
		sample
		for global_index, sample in enumerate(limited)
		if global_index % num_shards == shard_index
	)


def read_completed_question_ids(path: Path) -> Set[str]:
	"""Read completed identifiers from an append-only result JSONL file."""
	path = Path(path)
	if not path.exists():
		return set()
	completed: Set[str] = set()
	with path.open("r", encoding="utf-8") as source:
		for line_number, line in enumerate(source, start=1):
			if not line.strip():
				continue
			try:
				payload = json.loads(line)
				question_id = str(payload["question_id"])
			except (json.JSONDecodeError, KeyError, TypeError) as error:
				raise ValueError(
					f"invalid completed record at {path}:{line_number}: {error}",
				) from error
			if not question_id or question_id in completed:
				raise ValueError(
					f"empty or duplicate question_id at {path}:{line_number}",
				)
			completed.add(question_id)
	return completed


def append_records_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
	"""Append complete records only after checking all question IDs for conflicts."""
	path = Path(path)
	existing = read_completed_question_ids(path)
	new_question_ids = [str(record.get("question_id", "")) for record in records]
	if any(not question_id for question_id in new_question_ids):
		raise ValueError("every record must have a non-empty question_id")
	if len(new_question_ids) != len(set(new_question_ids)):
		raise ValueError("new records contain duplicate question_id values")
	conflicts = existing.intersection(new_question_ids)
	if conflicts:
		raise ValueError(f"refusing to append completed question IDs: {sorted(conflicts)}")
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as destination:
		for record in records:
			destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
		destination.flush()


def generate_mcts_records(
	samples: Sequence[SearchSample],
	executor: Any,
	cache: JsonlEvaluationCache,
	search_config: MCTSConfig,
	search_mode: SearchMode,
	model_id: str,
	model_revision: str,
	tokenizer_revision: str,
	batch_size: int,
	original_depth: int,
	score_generations: ScoreGenerations = score_math_generations,
	settings: GenerationSettings = GenerationSettings(),
	run_metadata: Optional[Mapping[str, Any]] = None,
) -> BatchedSearchResult:
	"""Search many questions and batch model calls that share a layer path."""
	_validate_inputs(samples=samples, batch_size=batch_size)
	search = ProgramMCTS(
		original_depth=original_depth,
		config=search_config,
		mode=search_mode,
	)
	active = [
		_ActiveSearch(
			sample=sample,
			session=search.start_session(),
			prompt_hash=prompt_sha256(sample.question),
		)
		for sample in samples
	]
	records: Dict[str, Dict[str, Any]] = {}
	score_cache: Dict[Tuple[str, str, str], EvaluationResult] = {}
	cache_hits = 0
	cache_misses = 0
	model_batches = 0
	generation_hash = settings.sha256()

	while len(records) < len(active):
		uncached_by_path: Dict[LayerPath, List[_ActiveSearch]] = {}
		for state in active:
			if state.sample.question_id in records:
				continue
			path = state.session.request_path()
			if path is None:
				records[state.sample.question_id] = _record_for_state(
					state=state,
					model_id=model_id,
					model_revision=model_revision,
					tokenizer_revision=tokenizer_revision,
					generation_hash=generation_hash,
					batch_size=batch_size,
					run_metadata=run_metadata or {},
				)
				continue

			identity = _cache_identity(
				sample=state.sample,
				prompt_hash=state.prompt_hash,
				path=path,
				model_id=model_id,
				model_revision=model_revision,
				tokenizer_revision=tokenizer_revision,
				generation_hash=generation_hash,
			)
			cached = cache.get(identity)
			if cached is not None:
				cache_hits += 1
				state.session.record_evaluation(
					EvaluationResult(
						binary_reward=cached.binary_reward,
						generated_answer=cached.generated_answer,
					),
				)
				continue
			cache_misses += 1
			uncached_by_path.setdefault(path, []).append(state)

		for path, path_states in uncached_by_path.items():
			for offset in range(0, len(path_states), batch_size):
				chunk = path_states[offset : offset + batch_size]
				generation = executor.generate_batch(
					[state.sample.question for state in chunk],
					path=path,
					settings=settings,
				)
				model_batches += 1
				generated_pairs = tuple(zip(
					chunk,
					generation.generated_answers,
					strict=True,
				))
				score_keys = tuple(
					(
						state.sample.question,
						state.sample.ground_truth,
						generated_answer,
					)
					for state, generated_answer in generated_pairs
				)
				pending_score_keys = tuple(dict.fromkeys(
					score_key
					for score_key in score_keys
					if score_key not in score_cache
				))
				if pending_score_keys:
					pending_evaluations = tuple(
						score_generations(pending_score_keys),
					)
					if len(pending_evaluations) != len(pending_score_keys):
						raise RuntimeError(
							"score_generations returned a different number of results",
						)
					for score_key, evaluation in zip(
						pending_score_keys,
						pending_evaluations,
						strict=True,
					):
						if evaluation.generated_answer != score_key[2]:
							raise RuntimeError(
								"score_generations changed generated-answer ordering",
							)
						score_cache[score_key] = evaluation

				for (state, generated_answer), score_key in zip(
					generated_pairs,
					score_keys,
					strict=True,
				):
					evaluation = score_cache[score_key]
					identity = _cache_identity(
						sample=state.sample,
						prompt_hash=state.prompt_hash,
						path=path,
						model_id=model_id,
						model_revision=model_revision,
						tokenizer_revision=tokenizer_revision,
						generation_hash=generation_hash,
					)
					cache.put(
						CachedEvaluation(
							identity=identity,
							binary_reward=evaluation.binary_reward,
							generated_answer=evaluation.generated_answer,
						),
					)
					state.session.record_evaluation(evaluation)

	return BatchedSearchResult(
		records=tuple(records[sample.question_id] for sample in samples),
		cache_hits=cache_hits,
		cache_misses=cache_misses,
		model_batches=model_batches,
	)


def _validate_inputs(samples: Sequence[SearchSample], batch_size: int) -> None:
	if not samples:
		raise ValueError("samples cannot be empty")
	if batch_size <= 0:
		raise ValueError("batch_size must be positive")
	question_ids = [sample.question_id for sample in samples]
	if len(question_ids) != len(set(question_ids)):
		raise ValueError("question_id values must be unique within a search batch")
	for sample in samples:
		if not sample.question_id or not sample.question or not sample.ground_truth:
			raise ValueError("question_id, question, and ground_truth must be non-empty")


def _cache_identity(
	sample: SearchSample,
	prompt_hash: str,
	path: LayerPath,
	model_id: str,
	model_revision: str,
	tokenizer_revision: str,
	generation_hash: str,
) -> CacheIdentity:
	return CacheIdentity(
		question_id=sample.question_id,
		path=path,
		model_id=model_id,
		model_revision=model_revision,
		tokenizer_revision=tokenizer_revision,
		prompt_hash=prompt_hash,
		generation_config_hash=generation_hash,
	)


def _record_for_state(
	state: _ActiveSearch,
	model_id: str,
	model_revision: str,
	tokenizer_revision: str,
	generation_hash: str,
	batch_size: int,
	run_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
	return search_result_record(
		question_id=state.sample.question_id,
		question=state.sample.question,
		ground_truth=state.sample.ground_truth,
		result=state.session.result(),
		additional_metadata={
			**state.sample.additional_metadata,
			**run_metadata,
			"model_id": model_id,
			"model_revision": model_revision,
			"tokenizer_revision": tokenizer_revision,
			"prompt_hash": state.prompt_hash,
			"generation_config_hash": generation_hash,
			"question_batch_size": batch_size,
		},
	)
