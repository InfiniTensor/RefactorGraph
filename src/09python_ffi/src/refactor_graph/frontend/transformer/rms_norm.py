from refactor_graph.frontend.modeling import InfiniTensorModel, DTYPE
import numpy as np


class RMSNorm(InfiniTensorModel):
    def __init__(self, hidden_size, eps: float = 1e-6, dtype=DTYPE.F32, **kwargs):
        super().__init__(**kwargs)
        self.eps = np.array(eps, dtype=np.float32)
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = self.parameter(
            np.ones(self.hidden_size, dtype=np.float32), "weight"
        )

    def __call__(self, hidden_states):
        super().__call__([hidden_states])
        if self.dtype != DTYPE.F32:
            hidden_states = self.cast(hidden_states, DTYPE.F32)
        variance = self.reduce_mean(
            self.pow(hidden_states, np.array(2, dtype=np.float32)), -1
        )
        hidden_states = self.mul(
            hidden_states,
            self.div(
                np.array(1, dtype=np.float32), self.sqrt(self.add(variance, self.eps))
            ),
        )
        hidden_states = self.mul(hidden_states, self.weight)
        if self.dtype != DTYPE.F32:
            hidden_states = self.cast(hidden_states, self.dtype)
        self.outputs = [hidden_states]
        return hidden_states
