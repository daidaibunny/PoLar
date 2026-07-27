import unittest
from types import SimpleNamespace

from reconstruction.evaluation import GenerationSettings, build_direct_prompt
from reconstruction.executor import (
	FrozenLayerPathExecutor,
	validate_execution_device,
	validate_frozen_model,
	validate_layer_path,
)


class FakeParameter:
	def __init__(self, requires_grad: bool = False) -> None:
		self.requires_grad = requires_grad


class FakeInputIds:
	def __init__(self, batch_size: int = 1, sequence_length: int = 3) -> None:
		self.shape = (batch_size, sequence_length)


class FakeInputs(dict):
	def __init__(self, batch_size: int = 1, sequence_length: int = 3) -> None:
		super().__init__(
			input_ids=FakeInputIds(
				batch_size=batch_size,
				sequence_length=sequence_length,
			),
		)
		self.destination = None

	def to(self, destination: str) -> "FakeInputs":
		self.destination = destination
		return self


class FakeTokenizer:
	def __init__(self) -> None:
		self.prompts = []

	def __call__(
		self,
		prompt,
		return_tensors: str,
		padding: bool = False,
	) -> FakeInputs:
		self.prompts.append((prompt, return_tensors, padding))
		batch_size = len(prompt) if isinstance(prompt, list) else 1
		sequence_length = 4 if isinstance(prompt, list) else 3
		return FakeInputs(batch_size=batch_size, sequence_length=sequence_length)

	@staticmethod
	def batch_decode(token_rows, skip_special_tokens: bool):
		if not skip_special_tokens:
			raise AssertionError("special tokens must be skipped")
		return [" \\boxed{" + str(row[-1]) + "} " for row in token_rows]


class FakeModel:
	device = "cpu"
	config = SimpleNamespace(num_hidden_layers=28)

	def __init__(self, requires_grad: bool = False) -> None:
		self._parameters = [FakeParameter(requires_grad)]
		self.generate_calls = []

	def parameters(self):
		return iter(self._parameters)

	def generate(self, **kwargs):
		self.generate_calls.append(kwargs)
		batch_size, sequence_length = kwargs["input_ids"].shape
		row_count = batch_size * kwargs.get("num_return_sequences", 1)
		return [
			list(range(sequence_length)) + [42 + row_index]
			for row_index in range(row_count)
		]


class ExecutorValidationTest(unittest.TestCase):
	def test_rejects_empty_non_integer_and_out_of_range_paths(self) -> None:
		with self.assertRaises(ValueError):
			validate_layer_path([], 28)
		with self.assertRaises(TypeError):
			validate_layer_path([0, "1"], 28)
		with self.assertRaises(ValueError):
			validate_layer_path([0, 28], 28)

	def test_requires_exact_depth_and_frozen_parameters(self) -> None:
		validate_frozen_model(FakeModel(), original_depth=28)
		with self.assertRaises(ValueError):
			validate_frozen_model(FakeModel(requires_grad=True), original_depth=28)
		wrong_depth = FakeModel()
		wrong_depth.config = SimpleNamespace(num_hidden_layers=27)
		with self.assertRaises(ValueError):
			validate_frozen_model(wrong_depth, original_depth=28)

	def test_never_silently_falls_back_from_requested_device(self) -> None:
		unavailable = SimpleNamespace(
			backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
			cuda=SimpleNamespace(is_available=lambda: False),
		)
		available = SimpleNamespace(
			backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
			cuda=SimpleNamespace(is_available=lambda: True),
		)

		with self.assertRaises(RuntimeError):
			validate_execution_device(unavailable, "mps")
		with self.assertRaises(RuntimeError):
			validate_execution_device(unavailable, "cuda")
		validate_execution_device(available, "mps")
		validate_execution_device(available, "cuda")
		validate_execution_device(unavailable, "cpu")
		with self.assertRaises(ValueError):
			validate_execution_device(available, "automatic")


class FrozenLayerPathExecutorTest(unittest.TestCase):
	def test_sets_every_path_and_decodes_only_new_tokens(self) -> None:
		model = FakeModel()
		tokenizer = FakeTokenizer()
		set_paths = []
		executor = FrozenLayerPathExecutor(
			model,
			tokenizer,
			path_setter=lambda target, path: set_paths.append((target, tuple(path))),
		)

		first = executor.generate(
			"What is one?",
			path=(0, 1, 3),
			settings=GenerationSettings(),
		)
		second = executor.generate("What is two?", path=(0, 2, 3))

		self.assertEqual(
			set_paths,
			[(model, (0, 1, 3)), (model, (0, 2, 3))],
		)
		self.assertEqual(first.generated_answers, ("\\boxed{42}",))
		self.assertEqual(second.generated_answers, ("\\boxed{42}",))
		self.assertEqual(tokenizer.prompts[0][0], build_direct_prompt("What is one?"))
		self.assertNotIn("system", tokenizer.prompts[0][0].lower())
		self.assertEqual(
			model.generate_calls[0]["max_new_tokens"],
			50,
		)
		self.assertFalse(model.generate_calls[0]["do_sample"])

	def test_batches_questions_with_one_generation_per_question(self) -> None:
		model = FakeModel()
		tokenizer = FakeTokenizer()
		set_paths = []
		executor = FrozenLayerPathExecutor(
			model,
			tokenizer,
			path_setter=lambda target, path: set_paths.append((target, tuple(path))),
		)

		result = executor.generate_batch(
			["What is one?", "What is two?"],
			path=(0, 1, 3),
			settings=GenerationSettings(),
		)

		self.assertEqual(set_paths, [(model, (0, 1, 3))])
		self.assertEqual(
			result.prompts,
			(
				build_direct_prompt("What is one?"),
				build_direct_prompt("What is two?"),
			),
		)
		self.assertEqual(result.generated_answers, ("\\boxed{42}", "\\boxed{43}"))
		self.assertEqual(tokenizer.prompts[0][1:], ("pt", True))
		self.assertEqual(model.generate_calls[0]["num_return_sequences"], 1)

	def test_reuses_identical_batched_model_inputs_across_layer_paths(self) -> None:
		model = FakeModel()
		tokenizer = FakeTokenizer()
		executor = FrozenLayerPathExecutor(
			model,
			tokenizer,
			path_setter=lambda target, path: None,
		)
		questions = ["What is one?", "What is two?"]

		executor.generate_batch(questions, path=(0, 1, 2))
		executor.generate_batch(questions, path=(0, 1, 1, 2))

		self.assertEqual(len(tokenizer.prompts), 1)
		self.assertIs(
			model.generate_calls[0]["input_ids"],
			model.generate_calls[1]["input_ids"],
		)

	def test_batch_generation_rejects_invalid_or_sampling_inputs(self) -> None:
		executor = FrozenLayerPathExecutor(FakeModel(), FakeTokenizer())

		with self.assertRaises(ValueError):
			executor.generate_batch([])
		with self.assertRaises(TypeError):
			executor.generate_batch(["Question", 3])
		with self.assertRaises(ValueError):
			executor.generate_batch(
				["Question"],
				settings=GenerationSettings(do_sample=True, num_return_sequences=2),
			)


if __name__ == "__main__":
	unittest.main()
