from refactor_graph.frontend.modeling import InfiniTensorModel, DTYPE
import numpy as np


class RMSNorm(InfiniTensorModel):
    def __init__(self, hidden_size, eps: float = 1e-6, dtype=DTYPE.F32, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = self.parameter(
            np.ones(self.hidden_size, dtype=self.dtype.np_type()), "weight"
        )

    def __call__(self, hidden_states):
        super().__call__([hidden_states])
        hidden_states = self.rms_norm(hidden_states, self.weight, self.eps)
        self.outputs = [hidden_states]
        return hidden_states
