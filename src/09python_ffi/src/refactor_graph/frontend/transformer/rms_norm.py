from refactor_graph.frontend.modeling import InfiniTensorModel, DTYPE
import numpy as np


class RMSNorm(InfiniTensorModel):
    def __init__(self, hidden_size, eps: float = 1e-6, dtype=DTYPE.F32, **kwargs):
        super().__init__(**kwargs)
        self.eps = np.array(eps, dtype=dtype.np_type())
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = self.parameter(
            np.ones(self.hidden_size, dtype=self.dtype.np_type()), "weight"
        )

    def __call__(self, hidden_states):
        super().__call__([hidden_states])
        variance = self.reduce_mean(
            self.pow(hidden_states, np.array(2, dtype=self.dtype.np_type())), -1
        )
        hidden_states = self.mul(
            hidden_states,
            self.div(
                np.array(1, dtype=self.dtype.np_type()), self.sqrt(self.add(variance, self.eps))
            ),
        )
        hidden_states = self.mul(hidden_states, self.weight)
        self.outputs = [hidden_states]
        return hidden_states
