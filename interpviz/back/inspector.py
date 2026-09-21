# inspector.py -- exports a pytorch model into a graph of FX nodes
# used by server.py: ModelInspector.graph() builds the graph, .forward() runs inference,
# .get_tensor() retrieves captured values for the popup display

# some notes on the TorchDynamo capture logic
# pytorch uses bytecode analysis to capture the model. including conditionals and loops.
# excluding normal python computation (only pytorch tensors are visible)
#
# torch.export.export(model, example_inputs) produces the FX graph:
#   ExportedProgram → GraphModule → Graph → [Node, Node, ...]
#
# FX = "effects" (shorthand like "special FX"). it's pytorch's symbolic tracing framework.
#
# each Node is one instruction with:
#   node.op     — type: "placeholder", "get_attr", "call_function", "call_method", "output"
#   node.name   — unique string like "linear", "relu", "add_1"
#   node.target — what to execute. for call_function: actual function (torch.ops.aten.mm.default).
#                 for get_attr: parameter path string ("layers.0.weight")
#   node.args   — tuple of inputs. references to other Node objects or literal values (scalars)
#   node.meta   — dict of metadata (shape, dtype, nn_module_stack, etc.)
#
# node types (node.op IS the type — same thing):
#   placeholder    — "read an input" → grabs from the args passed to Interpreter.run()
#   get_attr       — "read a weight/bias" → fetches parameter from the model
#   call_function  — "do math" → standalone function like torch.add(x, 3). looks up inputs from env, calls node.target
#   call_method    — tensor method like x.reshape(), x.add_(). torch.export rewrites most to call_function
#   output         — "we're done" → marks which nodes are final outputs
#
# Interpreter: pytorch's executor. calls run() → loops through every node → calls run_node() on each.
# self.env: {Node object: tensor value} — the live value dict. when a node runs, it looks up inputs from env,
# computes result, stores it back. that's how values flow between nodes.


import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.interpreter import Interpreter


# runs FX graph node-by-node, captures intermediate tensors, supports value overrides
# Interpreter is a pytorch class that re-executes an FX graph node by node
class CapturingInterpreter(Interpreter):

    # gm: the traced model pytorch module with a .graph member object attribute
    # marked: set of fx node names whose VALUES are kept (None = keep everything).
    #   shapes are always recorded (metadata is free) — unmarked tensors are dropped
    #   as soon as they run, so a big model can be visualized without OOMing.
    def __init__(self, gm: fx.GraphModule, overrides: dict[str, list] | None = None,
                 marked: set | None = None):
        # garbage_collect_values: off when capturing everything (old behavior);
        # on when selective — unneeded tensors are freed as the run proceeds
        super().__init__(gm, garbage_collect_values=marked is not None)
        # {fx_node_name: replacement_values} -- user-provided overrides for re-run. replacement_values is of type python list.
        self.overrides = overrides or {}
        self.marked = marked
        # {fx_node_name: torch.Tensor} -- captured during execution
        self.tensor_values = {}
        # {fx_node_name: [dims]} -- always recorded, marked or not
        self.tensor_shapes = {}

    def run_node(self, n: fx.Node):
        result = super().run_node(n)
        if n.name in self.overrides:
            result = torch.tensor(self.overrides[n.name], dtype=result.dtype, device=result.device)
        if isinstance(result, torch.Tensor):
            self.tensor_shapes[n.name] = list(result.shape)
            if self.marked is None or n.name in self.marked:
                self.tensor_values[n.name] = result.detach().clone()
        # output node returns a tuple wrapper; unwrap single-tensor returns so the output box shows values
        elif n.op == "output" and isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], torch.Tensor):
            self.tensor_shapes[n.name] = list(result[0].shape)
            if self.marked is None or n.name in self.marked:
                self.tensor_values[n.name] = result[0].detach().clone()
        # self.env is Interpreter's node object to tensor live value mapping {fx.Node object: tensor value}. we override it with our values.
        # we cannot just use self.env because it could be modified with in place ops so we still need self.tensor_values
        self.env[n] = result
        return result


def _round_val(v):
    # at most 2 decimal places, no trailing zeros (2.3 not 2.30)
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, list):
        return [_round_val(x) for x in v]
    return v


class ModelInspector:

    def __init__(self, model: nn.Module, example_inputs: tuple, dynamic_dims: list | None = None):
        self.model = model
        self.model.eval()
        # strict = False allows model branching in if statements
        # dynamic_dims: per-arg list of dims to keep dynamic (e.g. [[1]] = seq dim
        # of a (batch, seq) input) — without this the trace SPECIALIZES on the
        # example's exact shape and forward rejects any other size with
        # "Guard failed: x.size()[d] == N"
        dynamic_shapes = None
        if dynamic_dims:
            dynamic_shapes = tuple(
                {d: torch.export.Dim(f"dyn_{i}_{d}") for d in dims} if dims else None
                for i, dims in enumerate(dynamic_dims)
            )
        ep = torch.export.export(model, example_inputs, strict=False, dynamic_shapes=dynamic_shapes)
        self.gm = ep.module()

    def graph(self) -> dict:
        # builds graph: all FX nodes as-is (except _assert guards), direct edges from args
        # i.e. nodes = [{"name": "linear", "op": "call_function"}, {"name": "p_weight", "op": "get_attr"}, ...]
        # i.e. edges = [{"from": "p_weight", "to": "linear"}, {"from": "x", "to": "linear"}, ...]
        nodes = []
        edges = []
        for node in self.gm.graph.nodes:
            # torch.export inserts _assert, _guards_fn etc for shape/dtype guards — skip all _prefixed
            if node.name.startswith("_"):
                continue
            nodes.append({"name": node.name, "op": node.op})
            # edges from direct fx.Node references in args and kwargs
            for arg in node.args:
                if isinstance(arg, fx.Node) and not arg.name.startswith("_"):
                    edges.append({"from": arg.name, "to": node.name})
                # output node has nested tuple args like ((result_node,),)
                elif isinstance(arg, (tuple, list)):
                    for a in arg:
                        if isinstance(a, fx.Node) and not a.name.startswith("_"):
                            edges.append({"from": a.name, "to": node.name})
            # kwargs is a named dict parameter
            for v in node.kwargs.values():
                if isinstance(v, fx.Node) and not v.name.startswith("_"):
                    edges.append({"from": v.name, "to": node.name})
                elif isinstance(v, (tuple, list)):
                    for a in v:
                        if isinstance(a, fx.Node) and not a.name.startswith("_"):
                            edges.append({"from": a.name, "to": node.name})
        # drop zero-degree nodes: params/buffers nothing consumes (e.g. olmo's
        # per-layer rope_inv_freq — rope runs once globally, these are dead
        # checkpoint artifacts). they carry no information and clutter the board.
        connected = {e["from"] for e in edges} | {e["to"] for e in edges}
        nodes = [n for n in nodes if n["name"] in connected or n["op"] == "output"]
        return {"nodes": nodes, "edges": edges}

    def forward(self, *inputs, overrides: dict[str, list] | None = None, marked: set | None = None) -> dict:
        # marked: only these nodes' values are kept (shapes always recorded)
        interp = CapturingInterpreter(self.gm, overrides, marked)
        with torch.no_grad():
            output = interp.run(*inputs)
        self.captured_tensors = interp.tensor_values

        if isinstance(output, tuple):
            output_val = [o.detach().cpu().tolist() if isinstance(o, torch.Tensor) else o for o in output]
        elif isinstance(output, torch.Tensor):
            output_val = output.detach().cpu().tolist()
        else:
            output_val = output

        output_val = _round_val(output_val)
        # shapes keyed by FX name — covers every node, marked or not
        return {"output": output_val, "shapes": interp.tensor_shapes}

    def get_tensor(self, name: str, display: str = "all", k: int = 5) -> dict:
        # gets the values of the tensor when clicked
        # top k for feature analysis not used currently
        assert name in self.captured_tensors, f"no captured tensor for '{name}'"
        t = self.captured_tensors[name]
        # (numel,)
        shape = list(t.shape)

        if display == "topk":
            flat = t.flatten()
            actual_k = min(k, flat.numel())
            _, idxs = flat.abs().topk(actual_k)
            return {
                "name": name,
                "shape": shape,
                "display": "topk",
                "k": actual_k,
                "indices": idxs.cpu().tolist(),
                "values": _round_val(flat[idxs].cpu().tolist()),
            }
        return {
            "name": name,
            "shape": shape,
            "display": "all",
            "values": _round_val(t.cpu().tolist()),
        }
