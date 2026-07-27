"""Frozen direct-answer prompt and official DART-Math correctness scoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

from reconstruction.mcts import EvaluationResult


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ORIGINAL_DEPTH = 28
FULL_PATH = tuple(range(ORIGINAL_DEPTH))
MAX_NEW_TOKENS = 50
SAMPLING_TEMPERATURES = (0.3, 0.7, 1.0)
OFFICIAL_EVALUATOR_TIMEOUT_SECONDS = 60
OFFICIAL_EVALUATOR_PROCESSES = 4
MathScoreInput = Tuple[str, str, str]


@dataclass(frozen=True)
class GenerationSettings:
	"""Result-affecting text-generation settings used in cache identities."""

	max_new_tokens: int = MAX_NEW_TOKENS
	do_sample: bool = False
	temperature: Optional[float] = None
	top_p: Optional[float] = None
	num_return_sequences: int = 1

	def __post_init__(self) -> None:
		if self.max_new_tokens <= 0:
			raise ValueError("max_new_tokens must be positive")
		if self.num_return_sequences <= 0:
			raise ValueError("num_return_sequences must be positive")
		if not self.do_sample and (self.temperature is not None or self.top_p is not None):
			raise ValueError("Greedy generation cannot set temperature or top_p")

	def generate_kwargs(self) -> Dict[str, object]:
		"""Return explicit keyword arguments for ``model.generate``."""
		arguments: Dict[str, object] = {
			"max_new_tokens": self.max_new_tokens,
			"do_sample": self.do_sample,
			"num_return_sequences": self.num_return_sequences,
		}
		if self.temperature is not None:
			arguments["temperature"] = self.temperature
		if self.top_p is not None:
			arguments["top_p"] = self.top_p
		return arguments

	def sha256(self) -> str:
		"""Hash the canonical generation configuration."""
		return _canonical_sha256(asdict(self))


def build_direct_prompt(question: str) -> str:
	"""Build the paper's direct-answer prompt without chat or system wrappers."""
	return (
		"Solve the following math problem and output ONLY the final answer directly, "
		"formatted strictly as \\boxed{ANSWER}.\n"
		"### Problem Start\n"
		f"{question}\n"
		"### Problem End\n"
		"Answer:"
	)


def prompt_sha256(question: str) -> str:
	"""Hash the exact UTF-8 prompt supplied to the tokenizer."""
	return hashlib.sha256(build_direct_prompt(question).encode("utf-8")).hexdigest()


def sampling_settings(temperature: float, pass_k: int) -> GenerationSettings:
	"""Build one of the paper's explicitly reported sampling configurations."""
	if temperature not in SAMPLING_TEMPERATURES:
		raise ValueError(
			f"temperature must be one of {SAMPLING_TEMPERATURES}: {temperature}",
		)
	if pass_k not in range(1, 6):
		raise ValueError("pass_k must be between 1 and 5")
	return GenerationSettings(
		do_sample=True,
		temperature=temperature,
		num_return_sequences=pass_k,
	)


def score_math_generation(
	question: str,
	ground_truth: str,
	generated_answer: str,
	finish_reason: Optional[str] = None,
) -> EvaluationResult:
	"""Score one generation through the official PoLar batch evaluator."""
	if finish_reason in ("length", "abort"):
		return EvaluationResult(binary_reward=0, generated_answer=generated_answer)
	return score_math_generations(
		((question, ground_truth, generated_answer),),
	)[0]


def score_math_generations(
	score_inputs: Sequence[MathScoreInput],
) -> Tuple[EvaluationResult, ...]:
	"""Match official PoLar batch scoring, including its process timeout."""
	if not score_inputs:
		return ()
	try:
		from dart_math.eval import EvaluatorMathBatch
	except ImportError as error:
		raise RuntimeError(
			"The official DART-Math evaluator dependencies are not installed",
		) from error
	evaluator = EvaluatorMathBatch(
		strict_extract=True,
		use_orig_eq_for_olympiadbench=True,
		timeout=OFFICIAL_EVALUATOR_TIMEOUT_SECONDS,
	)
	samples = [
		SimpleNamespace(
			resp=generated_answer,
			ref_ans=ground_truth,
			ans=None,
			query="",
			dataset="math",
		)
		for _, ground_truth, generated_answer in score_inputs
	]
	_answers, corrects = evaluator.batch_eval(
		samples,
		n_procs=OFFICIAL_EVALUATOR_PROCESSES,
	)
	if len(corrects) != len(score_inputs):
		raise RuntimeError(
			"official evaluator returned a different number of scores than inputs",
		)
	return tuple(
		EvaluationResult(
			binary_reward=int(bool(correct)),
			generated_answer=generated_answer,
		)
		for (_, _, generated_answer), correct in zip(
			score_inputs,
			corrects,
			strict=True,
		)
	)


def _canonical_sha256(payload: Dict[str, object]) -> str:
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()
