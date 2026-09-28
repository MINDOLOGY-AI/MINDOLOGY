# OlMo2 1B for interpviz.
# Subclasses synapse's Olmo2ForCausalLM (instead of wrapping it) so traced FX
# node names stay single-pathed (model_layers_N_..., no double model_model).
# Loads the local HF safetensors weights via the evoke loader recipe
# (bf16, cpu). forward returns logits only, so the exported
# graph has a single output.

import torch
from safetensors.torch import load_file

from evoke.OlMo2_1b.run.config import OLMO2_1B_INSTRUCT_CONFIG
from evoke.OlMo2_1b.run.loader import WEIGHTS_DIR
from synapse.algorithms.transformer.olmo2 import Olmo2ForCausalLM

SNAPSHOT_DIR = WEIGHTS_DIR / "models--allenai--OLMo-2-0425-1B-Instruct" / "snapshots"


class Olmo1B(Olmo2ForCausalLM):
    def __init__(self):
        super().__init__(OLMO2_1B_INSTRUCT_CONFIG)
        # always bf16 on cpu: the traced graph (and the saved groups' node names) must not depend on the machine;
        # the server moves it to the gpu on demand
        dtype = torch.bfloat16
        self.to(dtype=dtype)

        # loader recipe: safetensors on cpu, cast to dtype, load_state_dict, eval
        safetensors_file = next(SNAPSHOT_DIR.rglob("model.safetensors"))
        state = load_file(safetensors_file, device="cpu")
        mapped = {k: (v.to(dtype) if v.dtype != dtype else v) for k, v in state.items()}
        self.load_state_dict(mapped)
        self.eval()

    def forward(self, input_ids):
        # drop loss and past_key_values: single-output graph
        return super().forward(input_ids)[0]
