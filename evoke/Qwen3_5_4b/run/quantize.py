"""
Swap nn.Linear layers for bitsandbytes int8 Linears (LLM.int8).

In-memory only: the checkpoint on disk is never modified. bnb quantizes each
weight when the module is moved to cuda; embeddings, norms, conv1d and the
tied lm_head stay in the original dtype.
"""

import torch.nn as nn
import bitsandbytes as bnb


def quantize_model_int8(model):
    """Replace every nn.Linear (except the tied lm_head) with Linear8bitLt."""
    # collect (parent, attr_name, module) first; mutating while walking breaks the iterator
    targets = []
    for name, module in model.named_modules():
        if name == "lm_head":
            continue
        if isinstance(module, nn.Linear):
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            targets.append((parent, attr, module))

    for parent, attr, linear in targets:
        int8_linear = bnb.nn.Linear8bitLt(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            has_fp16_weights=False,
        )
        # Int8Params is what actually quantizes on .to("cuda"); a plain
        # nn.Parameter would be moved to GPU unquantized
        int8_linear.weight = bnb.nn.Int8Params(
            linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False
        )
        if linear.bias is not None:
            int8_linear.bias = nn.Parameter(
                linear.bias.data.clone(), requires_grad=False
            )
        setattr(parent, attr, int8_linear)

    print(f"  quantized {len(targets)} Linear layers to int8")
    return model
