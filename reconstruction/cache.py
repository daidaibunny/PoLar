"""Append-only evaluation cache with complete experiment identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


class CacheFormatError(ValueError):
	"""Raised when an existing cache entry is malformed or conflicting."""


@dataclass(frozen=True)
class CacheIdentity:
	"""Every input that can change an evaluation result."""

	question_id: str
	path: Tuple[int, ...]
	model_id: str
	model_revision: str
	tokenizer_revision: str
	prompt_hash: str
	generation_config_hash: str


@dataclass(frozen=True)
class CachedEvaluation:
	"""A binary correctness observation for one question and one layer path."""

	identity: CacheIdentity
	binary_reward: int
	generated_answer: str


def make_cache_key(identity: CacheIdentity) -> str:
	"""Create a stable key from all result-affecting inputs."""
	payload = _identity_payload(identity)
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


class JsonlEvaluationCache:
	"""Load and append immutable evaluation records in JSON Lines format."""

	def __init__(self, path: Path) -> None:
		self.path = Path(path)
		self._entries: Dict[str, CachedEvaluation] = {}
		self._load()

	def get(self, identity: CacheIdentity) -> Optional[CachedEvaluation]:
		"""Return a cached evaluation when the complete identity matches."""
		return self._entries.get(make_cache_key(identity))

	def put(self, evaluation: CachedEvaluation) -> bool:
		"""Append a new evaluation; return false for an identical existing entry."""
		_validate_evaluation(evaluation)
		key = make_cache_key(evaluation.identity)
		previous = self._entries.get(key)
		if previous is not None:
			if previous != evaluation:
				raise CacheFormatError(f"Conflicting cache values for key={key}")
			return False

		self.path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"key": key,
			"identity": _identity_payload(evaluation.identity),
			"binary_reward": evaluation.binary_reward,
			"generated_answer": evaluation.generated_answer,
		}
		with self.path.open("a", encoding="utf-8") as destination:
			destination.write(json.dumps(payload, sort_keys=True) + "\n")
			destination.flush()
		self._entries[key] = evaluation
		return True

	def _load(self) -> None:
		if not self.path.exists():
			return
		with self.path.open("r", encoding="utf-8") as source:
			for line_number, line in enumerate(source, start=1):
				if not line.strip():
					continue
				try:
					payload = json.loads(line)
					evaluation = _evaluation_from_payload(payload)
					key = make_cache_key(evaluation.identity)
				except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
					raise CacheFormatError(
						f"Invalid cache entry at {self.path}:{line_number}: {error}",
					) from error
				if payload.get("key") != key:
					raise CacheFormatError(
						f"Cache key mismatch at {self.path}:{line_number}",
					)
				previous = self._entries.get(key)
				if previous is not None and previous != evaluation:
					raise CacheFormatError(
						f"Conflicting cache entries at {self.path}:{line_number}",
					)
				self._entries[key] = evaluation


def _identity_payload(identity: CacheIdentity) -> Dict[str, object]:
	payload = asdict(identity)
	payload["path"] = list(identity.path)
	return payload


def _evaluation_from_payload(payload: Dict[str, object]) -> CachedEvaluation:
	identity_payload = dict(payload["identity"])
	identity_payload["path"] = tuple(identity_payload["path"])
	evaluation = CachedEvaluation(
		identity=CacheIdentity(**identity_payload),
		binary_reward=int(payload["binary_reward"]),
		generated_answer=str(payload["generated_answer"]),
	)
	_validate_evaluation(evaluation)
	return evaluation


def _validate_evaluation(evaluation: CachedEvaluation) -> None:
	if evaluation.binary_reward not in (0, 1):
		raise ValueError("binary_reward must be exactly 0 or 1")

