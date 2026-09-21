'''
HookedModel: minimal activation-caching wrapper around an nn.Module.

subclasses define hook_points() -> {name: (module, is_pre)}.
is_pre=True registers a forward pre-hook (captures module INPUT),
is_pre=False registers a forward hook (captures module OUTPUT).

cache holds the last run only: acts are consumed per batch by the caller.
'''

import torch


class HookedModel:
    def __init__(self, model):
        self.model = model
        self.acts = {}
        self._points = self.hook_points()

    def hook_points(self):
        raise NotImplementedError

    def run(self, tokens, names):
        self.acts = {}
        handles = []
        for name in names:
            module, is_pre = self._points[name]
            if is_pre:
                def hook(module, inputs, name=name):
                    self.acts[name] = inputs[0]
                handles.append(module.register_forward_pre_hook(hook))
            else:
                def hook(module, inputs, output, name=name):
                    # some modules return tuples, the tensor is always first
                    self.acts[name] = output[0] if isinstance(output, tuple) else output
                handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad():
                logits, _, _ = self.model(input_ids=tokens)
        finally:
            for h in handles:
                h.remove()
        return logits
