#!/usr/bin/env python3
"""Verify that the official full layer path matches unpatched LLaMA execution."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from reconstruction.evaluation import (
	FULL_PATH,
	MODEL_ID,
	ORIGINAL_DEPTH,
	GenerationSettings,
	build_direct_prompt,
)
from reconstruction.executor import validate_execution_device, validate_frozen_model
from reconstruction.executor_validation import FullPathComparison, compare_full_path_results
from reconstruction.label_generation import load_search_samples
from scripts.generate_mcts_labels import resolve_model_revision


@dataclass(frozen=True)
class _ExecutionBatch:
	sequences: Any
	first_token_logits: Any
	layer_order: Tuple[int, ...]
	answers: Tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-file", type=Path, required=True)
	parser.add_argument("--split", choices=("train", "validation"), default="train")
	parser.add_argument("--output-json", type=Path, required=True)
	parser.add_argument("--model-id", default=MODEL_ID)
	parser.add_argument("--model-revision")
	parser.add_argument("--tokenizer-revision")
	parser.add_argument("--max-samples", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--max-new-tokens", type=int, default=50)
	parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
	arguments = parser.parse_args(argv)
	if arguments.max_samples <= 0:
		parser.error("--max-samples must be positive")
	if arguments.batch_size <= 0:
		parser.error("--batch-size must be positive")
	return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
	arguments = parse_args(argv)
	if arguments.output_json.exists():
		raise FileExistsError(f"refusing to overwrite validation: {arguments.output_json}")
	samples = load_search_samples(
		path=arguments.data_file,
		splits=(arguments.split,),
		max_samples=arguments.max_samples,
		num_shards=1,
		shard_index=0,
	)
	model_revision = resolve_model_revision(arguments.model_id, arguments.model_revision)
	tokenizer_revision = arguments.tokenizer_revision or model_revision
	model, tokenizer, torch_module = _load_unpatched_model(
		model_id=arguments.model_id,
		model_revision=model_revision,
		tokenizer_revision=tokenizer_revision,
		device=arguments.device,
	)
	settings = GenerationSettings(max_new_tokens=arguments.max_new_tokens)
	question_chunks = [
		samples[offset : offset + arguments.batch_size]
		for offset in range(0, len(samples), arguments.batch_size)
	]
	baseline_batches = tuple(
		_execute_batch(
			model=model,
			tokenizer=tokenizer,
			questions=[sample.question for sample in chunk],
			settings=settings,
			torch_module=torch_module,
		)
		for chunk in question_chunks
	)

	from llm_depth_router.model import apply_ulysses_patch, setup_custom_path

	apply_ulysses_patch(arguments.model_id)
	setup_custom_path(model, FULL_PATH)
	patched_batches = tuple(
		_execute_batch(
			model=model,
			tokenizer=tokenizer,
			questions=[sample.question for sample in chunk],
			settings=settings,
			torch_module=torch_module,
		)
		for chunk in question_chunks
	)
	comparisons = tuple(
		compare_full_path_results(
			baseline_sequences=baseline.sequences,
			patched_sequences=patched.sequences,
			baseline_logits=baseline.first_token_logits,
			patched_logits=patched.first_token_logits,
			baseline_layer_order=baseline.layer_order,
			patched_layer_order=patched.layer_order,
			expected_path=FULL_PATH,
			torch_module=torch_module,
		)
		for baseline, patched in zip(baseline_batches, patched_batches, strict=True)
	)
	passed = all(comparison.passed for comparison in comparisons)
	question_results = []
	for chunk, baseline, patched in zip(
		question_chunks,
		baseline_batches,
		patched_batches,
		strict=True,
	):
		for sample, baseline_answer, patched_answer in zip(
			chunk,
			baseline.answers,
			patched.answers,
			strict=True,
		):
			question_results.append(
				{
					"question_id": sample.question_id,
					"baseline_answer": baseline_answer,
					"patched_full_path_answer": patched_answer,
					"answers_identical": baseline_answer == patched_answer,
				},
			)
	payload = {
		"passed": passed,
		"method_claim": "independent reconstruction",
		"model_id": arguments.model_id,
		"model_revision": model_revision,
		"tokenizer_revision": tokenizer_revision,
		"full_path": list(FULL_PATH),
		"settings": asdict(settings),
		"batch_comparisons": [asdict(comparison) for comparison in comparisons],
		"questions": question_results,
		"peak_gpu_memory_bytes": _peak_gpu_memory_bytes(arguments.device, torch_module),
	}
	arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
	with arguments.output_json.open("x", encoding="utf-8") as destination:
		json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
		destination.write("\n")
	print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
	return 0 if passed else 2


def _load_unpatched_model(
	model_id: str,
	model_revision: str,
	tokenizer_revision: str,
	device: str,
) -> Tuple[Any, Any, Any]:
	try:
		import torch
		from transformers import AutoModelForCausalLM, AutoTokenizer
	except ImportError as error:
		raise RuntimeError("model execution dependencies are unavailable") from error

	validate_execution_device(torch, device)
	tokenizer = AutoTokenizer.from_pretrained(
		model_id,
		revision=tokenizer_revision,
		padding_side="left",
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model_arguments = {
		"revision": model_revision,
		"torch_dtype": torch.bfloat16,
		"low_cpu_mem_usage": True,
	}
	if device == "cuda":
		model_arguments["device_map"] = "cuda"
	model = AutoModelForCausalLM.from_pretrained(model_id, **model_arguments)
	if device != "cuda":
		model.to(device)
	model.eval()
	for parameter in model.parameters():
		parameter.requires_grad_(False)
	validate_frozen_model(model, original_depth=ORIGINAL_DEPTH)
	return model, tokenizer, torch


def _execute_batch(
	model: Any,
	tokenizer: Any,
	questions: Sequence[str],
	settings: GenerationSettings,
	torch_module: Any,
) -> _ExecutionBatch:
	prompts = [build_direct_prompt(question) for question in questions]
	model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
	layer_order = []
	handles = [
		layer.register_forward_hook(
			lambda _module, _inputs, _output, layer_index=layer_index: layer_order.append(
				layer_index,
			),
		)
		for layer_index, layer in enumerate(model.model.layers)
	]
	with torch_module.inference_mode():
		logits = model(**model_inputs, use_cache=False).logits[:, -1, :].detach().cpu()
	for handle in handles:
		handle.remove()
	with torch_module.inference_mode():
		sequences = model.generate(
			**model_inputs,
			**settings.generate_kwargs(),
			pad_token_id=tokenizer.pad_token_id,
		)
	input_length = model_inputs["input_ids"].shape[-1]
	new_token_ids = [row[input_length:] for row in sequences]
	answers = tuple(
		answer.strip()
		for answer in tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
	)
	return _ExecutionBatch(
		sequences=sequences.detach().cpu(),
		first_token_logits=logits,
		layer_order=tuple(layer_order),
		answers=answers,
	)


def _peak_gpu_memory_bytes(device: str, torch_module: Any) -> Optional[int]:
	if device != "cuda":
		return None
	return int(torch_module.cuda.max_memory_allocated())


if __name__ == "__main__":
	raise SystemExit(main())
