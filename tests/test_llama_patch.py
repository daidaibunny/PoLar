import unittest
from types import MethodType

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_depth_router.patches.llama import _llama_forward_patch


class LlamaPatchCacheTest(unittest.TestCase):
	def test_repeated_layer_uses_an_independent_generation_cache_slot(self) -> None:
		config = LlamaConfig(
			vocab_size=64,
			hidden_size=32,
			intermediate_size=64,
			num_hidden_layers=3,
			num_attention_heads=4,
			num_key_value_heads=2,
			max_position_embeddings=64,
			bos_token_id=1,
			eos_token_id=2,
			pad_token_id=0,
		)
		model = LlamaForCausalLM(config).eval()
		model.model.forward = MethodType(_llama_forward_patch, model.model)
		model.model.custom_path = [0, 1, 1, 2]
		input_ids = torch.tensor([[1, 3, 4], [0, 1, 5]])
		attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
		original_layer_indices = [layer.self_attn.layer_idx for layer in model.model.layers]

		with torch.no_grad():
			generated = model.generate(
				input_ids,
				attention_mask=attention_mask,
				max_new_tokens=2,
				do_sample=False,
			)

		self.assertEqual(generated.shape, (2, 5))
		self.assertEqual(
			[layer.self_attn.layer_idx for layer in model.model.layers],
			original_layer_indices,
		)


if __name__ == "__main__":
	unittest.main()
