from refactor_graph.frontend.modeling import InfiniTensorModel, DTYPE
import numpy as np


class Linear(InfiniTensorModel):
    """
    Linear layer follows the formula Y = XW + b, where W is weight of shape (in_features, out_features), b is
    optional bias of shape (out_features,). Input X should have shape (..., in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DTYPE = DTYPE.F32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        shape = (out_features, in_features)
        self.weight = self.parameter(
            (np.random.random(shape)).astype(dtype.np_type()), "weight"
        )
        self.use_bias = bias
        if self.use_bias:
            self.bias = self.parameter(
                np.random.random(out_features).astype(dtype.np_type()), "bias"
            )

    def __call__(self, input):
        super().__call__([input])
        output = self.matmul(input, self.weight, transB=1)
        if self.use_bias:
            output = self.add(output, self.bias)
        self.outputs.append(output)
        return output
