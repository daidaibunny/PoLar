"""Monte Carlo tree search over executable Transformer layer programs."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


LayerPath = Tuple[int, ...]


class SearchMode(str, Enum):
	"""Supported search spaces for diagnosis and predictor training."""

	DIAGNOSTIC = "diagnostic"
	PREDICTOR_COMPATIBLE = "predictor_compatible"


@dataclass(frozen=True, order=True)
class Action:
	"""Skip or repeat a contiguous block in the current execution path."""

	operation: str
	start: int
	block_length: int
	repeat_count: int = 0

	def __post_init__(self) -> None:
		if self.operation not in ("skip", "repeat"):
			raise ValueError(f"Unsupported operation: {self.operation!r}")
		if self.start < 0:
			raise ValueError("start must be non-negative")
		if self.block_length <= 0:
			raise ValueError("block_length must be positive")
		if self.operation == "skip" and self.repeat_count != 0:
			raise ValueError("skip actions cannot have a repeat_count")
		if self.operation == "repeat" and self.repeat_count <= 0:
			raise ValueError("repeat actions require a positive repeat_count")


@dataclass(frozen=True)
class EvaluationResult:
	"""Model output and exact binary reward for an executed path."""

	binary_reward: int
	generated_answer: str

	def __post_init__(self) -> None:
		if self.binary_reward not in (0, 1):
			raise ValueError("binary_reward must be exactly 0 or 1")


@dataclass(frozen=True)
class MCTSConfig:
	"""Search hyperparameters, including explicitly labelled source defaults."""

	n_simulations: int = 200
	exploration_constant: float = math.sqrt(2)
	length_penalty_lambda: float = 5.0
	random_action_probability: float = 0.1
	max_block_length: int = 4
	max_repeat_count_diagnostic: int = 4
	max_repeat_count_predictor: int = 1
	seed: int = 42

	def __post_init__(self) -> None:
		if self.n_simulations < 0:
			raise ValueError("n_simulations must be non-negative")
		if not 0.0 <= self.random_action_probability <= 1.0:
			raise ValueError("random_action_probability must be in [0, 1]")
		if self.exploration_constant < 0:
			raise ValueError("exploration_constant must be non-negative")
		if self.length_penalty_lambda < 0:
			raise ValueError("length_penalty_lambda must be non-negative")
		if self.max_block_length <= 0:
			raise ValueError("max_block_length must be positive")
		if self.max_repeat_count_diagnostic <= 0:
			raise ValueError("max_repeat_count_diagnostic must be positive")
		if self.max_repeat_count_predictor != 1:
			raise ValueError("max_repeat_count_predictor must be exactly one")


@dataclass
class Node:
	"""One unique program in the lazily expanded search tree."""

	path: LayerPath
	parent: Optional["Node"] = None
	action: Optional[Action] = None
	visits: int = 0
	reward_sum: float = 0.0
	unexpanded_actions: List[Action] = field(default_factory=list)
	children: Dict[Action, "Node"] = field(default_factory=dict)
	is_evaluated: bool = False
	binary_reward: Optional[int] = None
	generated_answer: Optional[str] = None


SearchNode = Node


@dataclass(frozen=True)
class SearchResult:
	"""Every evaluated transition and the metadata needed to audit the search."""

	initial_score: int
	valid_programs: Tuple[LayerPath, ...]
	invalid_programs: Tuple[LayerPath, ...]
	evaluations: Tuple[Tuple[LayerPath, EvaluationResult], ...]
	search_metadata: Dict[str, object]


def apply_action(path: LayerPath, action: Action) -> LayerPath:
	"""Apply an action to positions in the current execution path."""
	end = action.start + action.block_length
	if end > len(path):
		raise ValueError(f"Action {action!r} is out of bounds for path {path!r}")
	block = path[action.start:end]
	if action.operation == "skip":
		return path[: action.start] + path[end:]
	return path[: action.start] + block * (action.repeat_count + 1) + path[end:]


def enumerate_actions(
	path: LayerPath,
	mode: SearchMode,
	max_block_length: int,
	max_repeat_count: int,
	original_depth: int,
) -> Tuple[Action, ...]:
	"""Enumerate legal one-step edits, filtering to the selected search space."""
	actions: List[Action] = []
	maximum_block = min(max_block_length, len(path))
	for block_length in range(1, maximum_block + 1):
		for start in range(0, len(path) - block_length + 1):
			skip = Action("skip", start, block_length)
			skipped_path = apply_action(path, skip)
			if skipped_path and _mode_accepts(skipped_path, mode, original_depth):
				actions.append(skip)

			mode_repeat_limit = 1 if mode == SearchMode.PREDICTOR_COMPATIBLE else max_repeat_count
			for repeat_count in range(1, mode_repeat_limit + 1):
				repeat = Action("repeat", start, block_length, repeat_count)
				repeated_path = apply_action(path, repeat)
				if _mode_accepts(repeated_path, mode, original_depth):
					actions.append(repeat)
	return tuple(actions)


def is_predictor_compatible_path(path: LayerPath, original_depth: int) -> bool:
	"""Match the official skip/keep/repeat parser's segment language.

	The original layers are consumed from left to right in blocks of at most four.
	Each block is skipped, kept once, or executed exactly twice. This permits one
	extra execution but rejects arbitrary reordering and deeper repeats.
	"""
	if (
		not path
		or original_depth <= 0
		or any(layer < 0 or layer >= original_depth for layer in path)
	):
		return False
	memo: Dict[Tuple[int, int], bool] = {}

	def matches(original_position: int, path_position: int) -> bool:
		state = (original_position, path_position)
		if state in memo:
			return memo[state]
		if original_position == original_depth:
			return path_position == len(path)
		for block_length in range(1, min(4, original_depth - original_position) + 1):
			block = tuple(
				range(original_position, original_position + block_length),
			)
			if matches(original_position + block_length, path_position):
				memo[state] = True
				return True
			if path[path_position : path_position + block_length] == block:
				if matches(
					original_position + block_length,
					path_position + block_length,
				):
					memo[state] = True
					return True
			repeated = block + block
			if path[path_position : path_position + 2 * block_length] == repeated:
				if matches(
					original_position + block_length,
					path_position + 2 * block_length,
				):
					memo[state] = True
					return True
		memo[state] = False
		return False

	return matches(0, 0)


def ucb_score(
	reward_sum: float,
	visits: int,
	parent_visits: int,
	path_length: int,
	original_depth: int,
	exploration_constant: float,
	length_penalty_lambda: float,
) -> float:
	"""Score a child without modifying its stored binary reward."""
	if visits <= 0:
		return math.inf
	if original_depth <= 0:
		raise ValueError("original_depth must be positive")
	mean_reward = reward_sum / visits
	exploration = exploration_constant * math.sqrt(
		math.log(max(1, parent_visits)) / visits,
	)
	length_penalty = length_penalty_lambda * path_length / original_depth
	return mean_reward + exploration - length_penalty


class ProgramMCTS:
	"""Run seeded, lazy, globally deduplicated layer-program search."""

	def __init__(
		self,
		original_depth: int,
		config: MCTSConfig,
		mode: SearchMode,
	) -> None:
		if original_depth <= 0:
			raise ValueError("original_depth must be positive")
		self.original_depth = original_depth
		self.config = config
		self.mode = mode
		self._selection_random = random.Random(config.seed)

	def search(
		self,
		evaluator: Callable[[LayerPath], EvaluationResult],
	) -> SearchResult:
		"""Execute the base path and then up to ``n_simulations`` unique edits."""
		root_path = tuple(range(self.original_depth))
		root = self._make_node(root_path)
		seen = {root_path}
		evaluations: List[Tuple[LayerPath, EvaluationResult]] = []
		initial = self._evaluate(root, evaluator, evaluations)

		completed_simulations = 0
		for _ in range(self.config.n_simulations):
			node = root
			while True:
				child = self._expand_one(node, seen)
				if child is not None:
					result = self._evaluate(child, evaluator, evaluations)
					self._backpropagate(child, result.binary_reward)
					completed_simulations += 1
					break
				if not node.children:
					break
				node = self._select_child(node)

		valid = tuple(path for path, result in evaluations if result.binary_reward == 1)
		invalid = tuple(path for path, result in evaluations if result.binary_reward == 0)
		metadata: Dict[str, object] = {
			"mode": self.mode.value,
			"method_claim": "independent reconstruction",
			"n_simulations": self.config.n_simulations,
			"completed_simulations": completed_simulations,
			"seed": self.config.seed,
			"ucb_c": self.config.exploration_constant,
			"exploration_constant": self.config.exploration_constant,
			"exploration_constant_source": "independent reconstruction choice",
			"length_penalty_lambda": self.config.length_penalty_lambda,
			"length_penalty_source": "CoLa 2025 preliminary default",
			"random_exploration": self.config.random_action_probability,
			"random_action_probability": self.config.random_action_probability,
			"random_action_probability_source": "CoLa 2025 preliminary default",
			"simulation_count_source": "CoLa 2025 preliminary default",
			"ucb_parent_visit_definition": "immediate parent node visits",
			"simulation_definition": "one complete path execution with greedy decoding",
			"reward_definition": "binary DART-Math answer correctness",
			"max_block_length": self.config.max_block_length,
			"max_repeat_count": (
				self.config.max_repeat_count_predictor
				if self.mode == SearchMode.PREDICTOR_COMPATIBLE
				else self.config.max_repeat_count_diagnostic
			),
		}
		return SearchResult(
			initial_score=initial.binary_reward,
			valid_programs=valid,
			invalid_programs=invalid,
			evaluations=tuple(evaluations),
			search_metadata=metadata,
		)

	def _make_node(
		self,
		path: LayerPath,
		parent: Optional[Node] = None,
		action: Optional[Action] = None,
	) -> Node:
		repeat_limit = (
			self.config.max_repeat_count_predictor
			if self.mode == SearchMode.PREDICTOR_COMPATIBLE
			else self.config.max_repeat_count_diagnostic
		)
		actions = list(
			enumerate_actions(
				path=path,
				mode=self.mode,
				max_block_length=self.config.max_block_length,
				max_repeat_count=repeat_limit,
				original_depth=self.original_depth,
			),
		)
		seed_material = f"{self.config.seed}:{path}".encode("utf-8")
		shuffle_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
		random.Random(shuffle_seed).shuffle(actions)
		return Node(
			path=path,
			parent=parent,
			action=action,
			unexpanded_actions=actions,
		)

	def _expand_one(self, node: Node, seen: set[LayerPath]) -> Optional[Node]:
		while node.unexpanded_actions:
			action = node.unexpanded_actions.pop()
			path = apply_action(node.path, action)
			if path in seen:
				continue
			seen.add(path)
			child = self._make_node(path, parent=node, action=action)
			node.children[action] = child
			return child
		return None

	def _select_child(self, node: Node) -> Node:
		children = tuple(node.children.values())
		if self._selection_random.random() < self.config.random_action_probability:
			return self._selection_random.choice(children)
		return max(
			children,
			key=lambda child: (
				ucb_score(
					reward_sum=child.reward_sum,
					visits=child.visits,
					parent_visits=node.visits,
					path_length=len(child.path),
					original_depth=self.original_depth,
					exploration_constant=self.config.exploration_constant,
					length_penalty_lambda=self.config.length_penalty_lambda,
				),
				child.path,
			),
		)

	@staticmethod
	def _evaluate(
		node: Node,
		evaluator: Callable[[LayerPath], EvaluationResult],
		evaluations: List[Tuple[LayerPath, EvaluationResult]],
	) -> EvaluationResult:
		if node.is_evaluated:
			return EvaluationResult(
				binary_reward=int(node.binary_reward),
				generated_answer=str(node.generated_answer),
			)
		result = evaluator(node.path)
		node.is_evaluated = True
		node.binary_reward = result.binary_reward
		node.generated_answer = result.generated_answer
		evaluations.append((node.path, result))
		return result

	@staticmethod
	def _backpropagate(node: Node, reward: int) -> None:
		current: Optional[Node] = node
		while current is not None:
			current.visits += 1
			current.reward_sum += reward
			current = current.parent


def _mode_accepts(path: LayerPath, mode: SearchMode, original_depth: int) -> bool:
	if mode == SearchMode.DIAGNOSTIC:
		return True
	if mode == SearchMode.PREDICTOR_COMPATIBLE:
		return is_predictor_compatible_path(path, original_depth)
	raise ValueError(f"Unsupported search mode: {mode!r}")
