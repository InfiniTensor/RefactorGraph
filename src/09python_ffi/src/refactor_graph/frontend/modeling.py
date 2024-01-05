from math import inf
from python_ffi import (
    Compiler,
    Tensor,
    Operator,
    find_device,
    _make_data,
    _make_data_ex,
    _make_tensor,
    _make_compiler,
    _make_operator,
)
from typing import Dict, List, Any, Tuple, Union, Type
import numpy as np
from enum import Enum


class DTYPE(Enum):
    F32 = (1, np.float32)
    U8 = (2, np.uint8)
    I8 = (3, np.int8)
    U16 = (4, np.uint16)
    I16 = (5, np.int16)
    I32 = (6, np.int32)
    I64 = (7, np.int64)
    String = (8, np.string_)
    Bool = (9, np.bool_)
    FP16 = (10, np.float16)
    F64 = (11, np.float64)
    U32 = (12, np.uint32)
    U64 = (13, np.uint64)
    Complex64 = (14, np.complex64)
    Complex128 = (15, np.complex128)
    BF16 = (16, np.float32) # numpy does not support bf16 yet, should be handled by backend

    def onnx_type(self):
        return self.value[0]
    
    def np_type(self):
        return self.value[1]



def next_name(names_dict: Dict[str, int], name: str):
    if name in names_dict:
        result = f"{name}_{names_dict[name]}"
        names_dict[name] = names_dict[name] + 1
    else:
        result = f"{name}"
        names_dict[name] = 1
    return result


class InfiniTensorModel:
    """Base class for frontend modeling"""

    def __init__(
        self,
        model_name: str | None = None,
        _parameters: Dict[str, Tensor] | None = None,
        _const_edges: Dict[str, Tensor] | None = None,
        _nodes: Dict[str, Tuple[List[str], List[str]]] | None = None,
        _operators: Dict[str, Operator] | None = None,
        _node_names: Dict[str, int] | None = None,
        _tensor_names: Dict[str, int] | None = None,
    ) -> None:
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self._parameters: Dict[str, Tensor] = (
            _parameters if _parameters is not None else {}
        )
        self._const_edges: Dict[str, Tensor] = (
            _const_edges if _const_edges is not None else {}
        )
        self._nodes: Dict[str, Tuple[List[str], List[str]]] = (
            _nodes if _nodes is not None else {}
        )
        self._operators: Dict[str, Operator] = _operators if _operators is not None else {}

        # Fixed model name
        self.model_name = (
            model_name if model_name is not None else self.__class__.__name__
        )
        # The name prefix that can be updated
        self.base_name = self.model_name

        self.onnx_context = {"opset_version": 19}
        # Maps that keep track of naming uniqueness
        self._node_names: Dict[str, int] = _node_names if _node_names is not None else {}
        self._tensor_names: Dict[str, int] = (
            _tensor_names if _tensor_names is not None else {}
        )

        self._parent = None

    def __call__(self, inputs: List[str]) -> Union[str, None]:
        self.inputs = inputs.copy()
        return None

    def make_op(
        self,
        op_type: str,
        attributes: Dict[str, Any],
        inputs: Tuple[Union[str, np.ndarray], ...],
        outputs: Tuple[str, ...] | int = 1,
        name: str | None = None,
    ):
        # Get complete node name from op_type if name not given.
        # This node name should be unique throughout the whole model topology.
        node_name = (
            name
            if name is not None
            else next_name(self._node_names, f"{self.base_name}/{op_type}")
        )
        # generate output names
        if isinstance(outputs, int):
            output_names = [f"{node_name}_Output_{i}" for i in range(outputs)]
        else:
            output_names = [
                f"{node_name}_Output_{i}"
                if tensor_name is None or tensor_name == ""
                else tensor_name
                for i, tensor_name in enumerate(outputs)
            ]
        # process constant input
        input_names = []
        for i, input in enumerate(inputs):
            if isinstance(input, str):
                input_names.append(input)
            else:  # is numpy
                input_name = self.constant(input, f"{op_type}_Input")
                input_names.append(input_name)
        # make op
        self._operators[node_name] = _make_operator(
            self.onnx_context,
            op_type,
            attributes,
        )
        self._nodes[node_name] = (input_names, output_names)
        return output_names

    def make_tensor_np(self, data: np.ndarray, name: str):
        tensor_name = next_name(self._tensor_names, f"{self.base_name}/{name}")
        return tensor_name, _make_data(data)

    def parameter(self, data: np.ndarray, name: str = "Param"):
        tensor_name, tensor = self.make_tensor_np(data, name)
        self._parameters[tensor_name] = tensor
        return tensor_name

    def constant(self, data: np.ndarray, name: str = "Constant"):
        tensor_name, tensor = self.make_tensor_np(data, name)
        self._const_edges[tensor_name] = tensor
        return tensor_name

    def make_submodel(self, submodel_type, *args, **kwargs):
        kwargs["_parameters"] = self._parameters
        kwargs["_const_edges"] = self._const_edges
        kwargs["_nodes"] = self._nodes
        kwargs["_operators"] = self._operators
        kwargs["_node_names"] = self._node_names
        kwargs["_tensor_names"] = self._tensor_names
        kwargs["model_name"] = next_name(
            self._node_names, f"{self.base_name}/{submodel_type.__name__}"
        )
        submodel = submodel_type(*args, **kwargs)
        submodel.parent = self
        return submodel

    def make_compiler(self, inputs: Dict[str, Tuple[DTYPE, List[Any]]]):
        edges: Dict[str, Tensor] = {}
        for input in self.inputs:
            data = inputs.get(input)
            assert data is not None, f"Input [{input}] not provided!"
            edges[input] = _make_tensor(data[0].onnx_type(), data[1])
        edges.update(self._const_edges)
        edges.update(self._parameters)
        return _make_compiler(
            self._nodes, self._operators, edges, self.inputs, self.outputs
        )

    def load_param(self, data: Dict[str, np.ndarray]):
        for name in self._parameters:
            new_data = data.get(name)
            if new_data is not None:
                self._parameters[name] = _make_data(new_data)
            else:
                print(f"Warning: Value for {name} is not provided for loading.")


    #############################
    # Operator APIs
    #############################
    def sqrt(self, X, Y=""):
        return self.make_op("Sqrt", {}, (X,), (Y,))[0]

    def sigmoid(self, X, Y=""):            
        return self.make_op("Sigmoid", {}, (X,), (Y,))[0]

    def add(self, A, B, C=""):
        return self.make_op("Add", {}, (A, B), (C,))[0]
    
    def sub(self, A, B, C=""):
        return self.make_op("Sub", {}, (A, B), (C,))[0]
    
    def mul(self, A, B, C=""):
        return self.make_op("Mul", {}, (A, B), (C,))[0]
    
    def div(self, A, B, C=""):
        return self.make_op("Div", {}, (A, B), (C,))[0]
    
    def pow(self, A, B, C=""):
        return self.make_op("Pow", {}, (A, B), (C,))[0]
    
    def matmul(self, A, B, Y=""):
        return self.make_op("MatMul", {}, (A, B), (Y,))[0]
        
    def gemm(self, A, B, C=None, Y="", alpha=1.0, beta=1.0, transA=0, transB=0):
        inputs = (A, B, C) if C is not None else (A, B)
        return self.make_op("Gemm", {"alpha":alpha, "beta":beta, "transA":transA, "transB":transB}, inputs, (Y,))[0]

    def reshape(self, data, shape, reshaped=""):
        return self.make_op("Reshape", {}, (data, shape), (reshaped,))[0]
    
    def transpose(self, data, perm: List[int], transposed=""):
        return self.make_op("Transpose", {"perm": perm}, (data,), (transposed,))[0]
    
    def squeeze(self, data, axes: int | List[int], squeezed=""):
        return self.make_op("Squeeze", {}, (data, np.array(axes)), (squeezed,))[0]

    def unsqueeze(self, data, axes: int | List[int], unsqueezed=""):
        return self.make_op("Unsqueeze", {}, (data, np.array(axes)), (unsqueezed,))[0]
    
    def gather(self, data, indices, axis, output=""):
        return self.make_op("Gather", {"axis": axis}, (data, indices), (output,))[0]

    #TODO: support array inputs in the future
    def slice(self, data, axis, start=0, end=9223372036854775807, step = 1):
        return self.make_op("Slice", {}, (data, np.array(start), np.array(end), np.array(axis), np.array(step)))[0]

    def concat(self, inputs: Tuple[str,...], axis, result=""):
        return self.make_op("Concat", {"axis": axis}, inputs, (result,))[0]

    def split(self, input, axis, num_outputs):
        outputs = self.make_op("Split", {"axis": axis, "num_outputs": num_outputs}, (input,), num_outputs)
        return tuple(outputs)

    def reduce_sum(self, data, axes: List[int], reduced="", keepdims=1):
        return self.make_op("ReduceSum", {"keepdims": keepdims}, (data, np.array(axes)), (reduced,))[0]

    def cast(self, input, to: DTYPE, output=""):
        return self.make_op("Cast", {"to": to.onnx_type()}, (input,), (output,))
    
