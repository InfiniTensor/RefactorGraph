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
from enum import IntEnum


class DTYPE(IntEnum):
    F32 = 1
    U8 = 2
    I8 = 3
    U16 = 4
    I16 = 5
    I32 = 6
    I64 = 7
    String = 8
    Bool = 9
    FP16 = 10
    F64 = 11
    U32 = 12
    U64 = 13
    Complex64 = 14
    Complex128 = 15
    BF16 = 16


def next_name(names_dict: Dict[str, int], name: str):
    if name in names_dict:
        result = f"{name}_{names_dict[name]}"
        names_dict[name] = names_dict[name] + 1
    else:
        result = f"{name}"
        names_dict[name] = 1
    return result


class InfiniTensorModel:
    def __init__(
        self,
        model_name: str | None = None,
        parameters: Dict[str, Tensor] | None = None,
        const_edges: Dict[str, Tensor] | None = None,
        nodes: Dict[str, Tuple[List[str], List[str]]] | None = None,
        operators: Dict[str, Operator] | None = None,
        node_names: Dict[str, int] | None = None,
        tensor_names: Dict[str, int] | None = None,
    ) -> None:
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.parameters: Dict[str, Tensor] = (
            parameters if parameters is not None else {}
        )
        self.const_edges: Dict[str, Tensor] = (
            const_edges if const_edges is not None else {}
        )
        self.nodes: Dict[str, Tuple[List[str], List[str]]] = (
            nodes if nodes is not None else {}
        )
        self.operators: Dict[str, Operator] = operators if operators is not None else {}

        # Fixed model name
        self.model_name = (
            model_name if model_name is not None else self.__class__.__name__
        )
        # The name prefix that can be updated 
        self.base_name = self.model_name

        self.onnx_context = {"opset_version": 19}
        # Maps that keep track of naming uniqueness
        self.node_names: Dict[str, int] = node_names if node_names is not None else {}
        self.tensor_names: Dict[str, int] = tensor_names if tensor_names is not None else {}

        self.parent = None

    def __call__(self, inputs: List[str]) -> Union[str, None]:
        self.inputs = inputs.copy()
        if self.parent != None:
            self.base_name = next_name(self.parent.node_names, self.model_name)
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
            else next_name(self.node_names, f"{self.base_name}/{op_type}")
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
                input_name = self.constant(input, f"{node_name}_Input")
                input_names.append(input_name)
        # make op
        self.operators[node_name] = _make_operator(
            self.onnx_context,
            op_type,
            attributes,
        )
        self.nodes[node_name] = (input_names, output_names)
        return output_names

    def make_tensor_np(self, data: np.ndarray, name: str):
        tensor_name = next_name(self.tensor_names, name)
        return tensor_name, _make_data(data)

    def parameter(self, data: np.ndarray, name: str = "Param"):
        tensor_name, tensor = self.make_tensor_np(data, name)
        self.parameters[tensor_name] = tensor
        return tensor_name

    def constant(self, data: np.ndarray, name: str = "Constant"):
        tensor_name, tensor = self.make_tensor_np(data, name)
        self.const_edges[tensor_name] = tensor
        return tensor_name

    def make_submodel(self, submodel_type, *args, **kwargs):
        kwargs["parameters"] = self.parameters
        kwargs["const_edges"] = self.const_edges
        kwargs["nodes"] = self.nodes
        kwargs["operators"] = self.operators
        kwargs["node_names"] = self.node_names
        kwargs["tensor_names"] = self.tensor_names
        submodel = submodel_type(*args, **kwargs)
        submodel.model_name = f"{self.base_name}/{submodel.model_name}"
        submodel.base_name = submodel.model_name
        submodel.parent = self
        return submodel

    def make_compiler(self, inputs: Dict[str, Tuple[DTYPE, List[Any]]]):
        edges: Dict[str, Tensor] = {}
        for input in self.inputs:
            data = inputs.get(input)
            assert data is not None, f"Input [{input}] not provided!"
            edges[input] = _make_tensor(data[0].value, data[1])
        edges.update(self.const_edges)
        edges.update(self.parameters)
        return _make_compiler(
            self.nodes, self.operators, edges, self.inputs, self.outputs
        )

    def load_param(self, data: Dict[str, np.ndarray]):
        for name in self.parameters:
            new_data = data.get(name)
            if new_data is not None:
                self.parameters[name] = _make_data(new_data)
            else:
                print(f"Warning: Value for {name} is not provided for loading.")

    #############################
    # Operator APIs
    #############################
