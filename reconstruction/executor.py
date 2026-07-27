"""Frozen LLaMA execution through explicit, validated layer-index paths."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from reconstruction.evaluation import (
	FULL_PATH,
	MODEL_ID,
	ORIGINAL_DEPTH,
	GenerationSettings,
	build_direct_prompt,
)
from reconstruction.mcts import LayerPath


@dataclass(frozen=True)
class GenerationBatch:
	"""Decoded suffixes from one prompt and one layer path."""

	prompt: str
	path: LayerPath
	generated_answers: Tuple[str, ...]


def validate_layer_path(path: Sequence[int], original_depth: int) -> LayerPath:
	"""Return a non-empty in-range integer path suitable for the official patch."""
	if original_depth <= 0:
		raise ValueError("original_depth must be positive")
	if not path:
		raise ValueError("layer path cannot be empty")
	if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in path):
		raise TypeError("every layer index must be an integer")
	canonical_path = tuple(path)
	out_of_range = [
		layer for layer in canonical_path if layer < 0 or layer >= original_depth
	]
	if out_of_range:
		raise ValueError(
			f"layer path contains out-of-range indices for depth {original_depth}: "
			f"{out_of_range}",
		)
	return canonical_path


def validate_frozen_model(model: Any, original_depth: int = ORIGINAL_DEPTH) -> None:
	"""Verify architecture depth and ensure every model parameter is frozen."""
	configured_depth = getattr(getattr(model, "config", None), "num_hidden_layers", None)
	if configured_depth != original_depth:
		raise ValueError(
			f"Expected {original_depth} hidden layers, found {configured_depth!r}",
		)
	unfrozen = sum(bool(getattr(parameter, "requires_grad", False)) for parameter in model.parameters())
	if unfrozen:
		raise ValueError(f"Model has {unfrozen} parameters with requires_grad=True")


class FrozenLayerPathExecutor:
	"""Generate direct answers after setting an explicit path on a frozen model."""

	def __init__(
		self,
		model: Any,
		tokenizer: Any,
		original_depth: int = ORIGINAL_DEPTH,
		path_setter: Optional[Callable[[Any, Sequence[int]], None]] = None,
	) -> None:
		validate_frozen_model(model, original_depth)
		self.model = model
		self.tokenizer = tokenizer
		self.original_depth = original_depth
		self.path_setter = path_setter or _official_path_setter

	def generate(
		self,
		question: str,
		path: Sequence[int] = FULL_PATH,
		settings: GenerationSettings = GenerationSettings(),
	) -> GenerationBatch:
		"""Set the path, generate, and decode only newly generated token IDs."""
		canonical_path = validate_layer_path(path, self.original_depth)
		self.path_setter(self.model, canonical_path)
		prompt = build_direct_prompt(question)
		model_inputs = self.tokenizer(prompt, return_tensors="pt")
		model_inputs = model_inputs.to(self.model.device)
		input_length = model_inputs["input_ids"].shape[-1]

		with _inference_context():
			generated_ids = self.model.generate(
				**model_inputs,
				**settings.generate_kwargs(),
			)
		new_token_ids = [output_ids[input_length:] for output_ids in generated_ids]
		answers = tuple(
			answer.strip()
			for answer in self.tokenizer.batch_decode(
				new_token_ids,
				skip_special_tokens=True,
			)
		)
		return GenerationBatch(
			prompt=prompt,
			path=canonical_path,
			generated_answers=answers,
		)

	def final_prompt_logits(
		self,
		question: str,
		path: Sequence[int] = FULL_PATH,
	) -> Any:
		"""Return logits for the first generated token equivalence check."""
		canonical_path = validate_layer_path(path, self.original_depth)
		self.path_setter(self.model, canonical_path)
		model_inputs = self.tokenizer(build_direct_prompt(question), return_tensors="pt")
		model_inputs = model_inputs.to(self.model.device)
		with _inference_context():
			outputs = self.model(**model_inputs, use_cache=False)
		return outputs.logits[:, -1, :]


def load_frozen_llama_executor(
	model_id: str = MODEL_ID,
	model_revision: Optional[str] = None,
	tokenizer_revision: Optional[str] = None,
	device_map: str = "cuda",
) -> FrozenLayerPathExecutor:
	"""Load the requested model in bfloat16 and apply the official LLaMA patch."""
	try:
		import torch
		from transformers import AutoModelForCausalLM, AutoTokenizer
		from llm_depth_router.model import apply_ulysses_patch
	except ImportError as error:
		raise RuntimeError("Official model execution dependencies are unavailable") from error

	apply_ulysses_patch(model_id)
	tokenizer = AutoTokenizer.from_pretrained(
		model_id,
		revision=tokenizer_revision or model_revision,
		padding_side="left",
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		revision=model_revision,
		torch_dtype=torch.bfloat16,
		device_map=device_map,
	)
	model.eval()
	for parameter in model.parameters():
		parameter.requires_grad_(False)
	return FrozenLayerPathExecutor(model=model, tokenizer=tokenizer)


def _official_path_setter(model: Any, path: Sequence[int]) -> None:
	from llm_depth_router.model import setup_custom_path

	setup_custom_path(model, list(path))


def _inference_context() -> Any:
	try:
		import torch
	except ImportError:
		return nullcontext()
	return torch.inference_mode()
