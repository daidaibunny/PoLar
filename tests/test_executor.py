import unittest
from types import SimpleNamespace

from reconstruction.evaluation import GenerationSettings, build_direct_prompt
from reconstruction.executor import (
	FrozenLayerPathExecutor,
	validate_frozen_model,
	validate_layer_path,
)


class FakeParameter:
	def __init__(self, requires_grad: bool = False) -> None:
		self.requires_grad = requires_grad


class FakeInputIds:
	shape = (1, 3)


class FakeInputs(dict):
	def __init__(self) -> None:
		super().__init__(input_ids=FakeInputIds())
		self.destination = None

	def to(self, destination: str) -> "FakeInputs":
		self.destination = destination
		return self


class FakeTokenizer:
	def __init__(self) -> None:
		self.prompts = []

	def __call__(self, prompt: str, return_tensors: str) -> FakeInputs:
		self.prompts.append((prompt, return_tensors))
		return FakeInputs()

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
		return [
			[10, 11, 12, 42],
			[10, 11, 12, 43],
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
		self.assertEqual(first.generated_answers, ("\\boxed{42}", "\\boxed{43}"))
		self.assertEqual(second.generated_answers, ("\\boxed{42}", "\\boxed{43}"))
		self.assertEqual(tokenizer.prompts[0][0], build_direct_prompt("What is one?"))
		self.assertNotIn("system", tokenizer.prompts[0][0].lower())
		self.assertEqual(
			model.generate_calls[0]["max_new_tokens"],
			50,
		)
		self.assertFalse(model.generate_calls[0]["do_sample"])


if __name__ == "__main__":
	unittest.main()
