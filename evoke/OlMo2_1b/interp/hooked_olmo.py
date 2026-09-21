'''
HookedOlmo: hook point definitions for the synapse Olmo2.

post_gate is a PRE-hook on down_proj: SiLU(gate)*up is computed inline
inside GatedMLP.forward and only exists as down_proj's input.
everything else is a normal forward hook on a module output.
'''

from synapse.interp.hooked_model import HookedModel


class HookedOlmo(HookedModel):
    def hook_points(self):
        core = self.model.model
        points = {"embed": (core.embed_tokens, False)}
        for i, layer in enumerate(core.layers):
            points[f"attn_out.{i}"] = (layer.self_attn, False)
            points[f"post_gate.{i}"] = (layer.mlp.down_proj, True)
            points[f"mlp_out.{i}"] = (layer.mlp, False)
            points[f"resid_post.{i}"] = (layer, False)
        points["final_norm"] = (core.norm, False)
        return points
