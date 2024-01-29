from refactor_graph.frontend.modeling import InfiniTensorModel, DTYPE
from refactor_graph.frontend.nn import Linear


class FeedForward(InfiniTensorModel):
    def __init__(self, hidden_size, intermediate_size, dtype=DTYPE.F32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.gate_proj = self.make_submodel(
            Linear, self.hidden_size, self.intermediate_size, False, dtype, model_name = "gate_proj"
        )
        self.up_proj = self.make_submodel(
            Linear, self.hidden_size, self.intermediate_size, False, dtype, model_name = "up_proj"
        )
        self.down_proj = self.make_submodel(
            Linear, self.intermediate_size, self.hidden_size, False, dtype, model_name = "down_proj"
        )
        self.act_fn = self.silu

    def __call__(self, x):
        super().__call__([x])
        output = self.down_proj(
            self.mul(self.act_fn(self.gate_proj(x)), self.up_proj(x))
        )
        self.outputs = [output]
        return output
