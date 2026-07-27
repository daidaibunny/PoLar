"""Frozen direct-answer prompt and official DART-Math correctness scoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, Optional

from reconstruction.mcts import EvaluationResult


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ORIGINAL_DEPTH = 28
FULL_PATH = tuple(range(ORIGINAL_DEPTH))
MAX_NEW_TOKENS = 50
SAMPLING_TEMPERATURES = (0.3, 0.7, 1.0)


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
	"""Score a boxed generation with the unmodified DART-Math evaluator."""
	if "oxed{" not in generated_answer:
		return EvaluationResult(binary_reward=0, generated_answer=generated_answer)
	try:
		from dart_math.eval import EvaluatorMath
	except ImportError as error:
		raise RuntimeError(
			"The official DART-Math evaluator dependencies are not installed",
		) from error

	evaluator = EvaluatorMath(
		strict_extract=True,
		use_orig_eq_for_olympiadbench=True,
	)
	sample = SimpleNamespace(
		resp=generated_answer,
		ref_ans=ground_truth,
		ans=None,
		query=question,
		dataset="math",
		finish_reason=finish_reason,
	)
	try:
		correct = bool(evaluator.eval(sample))
	except ValueError:
		# Malformed model output is an ordinary negative reward, not a run failure.
		correct = False
	return EvaluationResult(
		binary_reward=int(correct),
		generated_answer=generated_answer,
	)


def _canonical_sha256(payload: Dict[str, object]) -> str:
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()
