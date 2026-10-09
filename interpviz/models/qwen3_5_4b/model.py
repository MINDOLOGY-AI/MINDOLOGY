# Qwen3.5-4B (text) for interpviz.
# Subclasses synapse's Qwen3_5ForCausalLM (instead of wrapping it) so module paths and traced FX node names stay
# single-pathed (model.language_model.layers.N). Weights come from the evoke loader (bf16, cpu: the traced graph and
# the saved groups' node names must not depend on the machine; the server moves it to the gpu on demand).
# forward returns logits only, so the exported graph has a single output.

import torch

from evoke.Qwen3_5_4b.run.config import QWEN3_5_4B_CONFIG
from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_model
from synapse.algorithms.transformer.qwen3_5 import Qwen3_5ForCausalLM


class Qwen4B(Qwen3_5ForCausalLM):
    def __init__(self):
        super().__init__(QWEN3_5_4B_CONFIG)
        dtype = torch.bfloat16
        self.to(dtype=dtype)
        loaded, _ = load_qwen3_5_model(device=torch.device("cpu"), dtype=dtype)
        self.load_state_dict(loaded.state_dict())
        self.tie_weights()
        self.eval()

    def forward(self, input_ids):
        # no kv / recurrent-state cache, drop loss and cache: single-output graph
        return super().forward(input_ids, use_cache=False)[0]
