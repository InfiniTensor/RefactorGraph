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
from collections import OrderedDict


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
    BF16 = (
        16,
        np.float16,
    )  # TODO:numpy does not support bf16 yet, should be handled by backend

    def onnx_type(self):
        return self.value[0]

    def np_type(self):
        return self.value[1]


def find_onnx_type(numpy_type):
    for dtype in DTYPE:
        if dtype.np_type() == numpy_type:
            return dtype.onnx_type()
    raise ValueError(f"No corresponding ONNX type found for numpy type: {numpy_type}")


def next_name(names_dict: Dict[str, int], name: str):
    if name in names_dict:
        result = f"{name}_{names_dict[name]}"
        names_dict[name] = names_dict[name] + 1
    else:
        result = f"{name}"
        names_dict[name] = 1
    return result


def resolve_variable(
    data: Tuple[Union[str, int], ...] | List[Union[str, int]],
    variable_map: Dict[str, int],
):
    result = []
    for item in data:
        if isinstance(item, int):
            result.append(item)
        else:
            result.append(variable_map[item])
    return result


class InfiniTensorModel:
    """
    Base class for frontend modeling

    Users can define their own models like this:

        class MyModel(InfiniTensorModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.weight = self.parameter(np.array(...), "Weight")

            def __call__(self, input1, input2):
                super().__call__([input1, input2])
                output = self.add(input1, input2)
                output = self.matmal(output, self.weight)
                self.outputs = [output]
                return output
    """

    def __init__(
        self,
        model_name: str | None = None,
        _parameters: OrderedDict[str, np.ndarray] | None = None,
        _const_edges: Dict[str, np.ndarray] | None = None,
        _nodes: Dict[str, Tuple[List[str], List[str]]] | None = None,
        _operators: Dict[str, Tuple[str, Dict[str, Any]]] | None = None,
        _node_names: Dict[str, int] | None = None,
        _tensor_names: Dict[str, int] | None = None,
        _dynamic_tensors: Dict[str, Tuple[Tuple[Union[str, int], ...], DTYPE]]
        | None = None,
        _cache: List[Tuple[str, str, Any]] | None = None,
    ) -> None:
        """The initializer for InfiniTensor Model.

        Note: when creating a submodel inside a model, self.make_submodel should be used instead of calling initializer
              directly to guarantee topo and naming inheritance.

        Args:
            model_name: Name of the model, default to class name.

        The following arguments are shared among submodels, and are updated implicitly. They should not be touched by
        end users, and it is important for a user-defined model to leave **kwargs in their initializer args for these
        arguments to be passed normally.
            _parameters ({tensor_name: np.ndarray}): Trainable parameters.
            _const_edges ({tensor_name: np.ndarray}): Constants.
            _nodes ({node_name: (inputs, outputs)}): Topology of op nodes and edges
            _operators ({node_name: op}): Operator nodes
            _node_names ({node_name: int}): Counter for seen node names, used for naming uniqueness
            _tensor_names ({tensor_name: int}): Counter for seen tensor names, used for naming uniqueness
            _dynamic_tensors ({tensor_name: (int|str,...)}): Dynamic tensors containing number or str
            _cache ({tensor_name: Memory | None}): cached tensors that live across multiple runs
        """
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self._parameters: OrderedDict[str, np.ndarray] = (
            _parameters if _parameters is not None else OrderedDict()
        )
        self._const_edges: Dict[str, np.ndarray] = (
            _const_edges if _const_edges is not None else {}
        )
        self._nodes: Dict[str, Tuple[List[str], List[str]]] = (
            _nodes if _nodes is not None else {}
        )
        self._operators: Dict[str, Tuple[str, Dict[str, Any]]] = (
            _operators if _operators is not None else {}
        )
        self._dynamic_tensors: Dict[str, Tuple[Tuple[Union[str, int], ...], DTYPE]] = (
            _dynamic_tensors if _dynamic_tensors is not None else {}
        )
        self._cache: List[Tuple[str, str, Any]] = _cache if _cache is not None else []
        # Fixed model name, default to class name
        self.model_name = (
            model_name if model_name is not None else self.__class__.__name__
        )
        # The name prefix that can be updated
        self.base_name = self.model_name
        # TODO: the current backend is following onnx opset standard, and version number
        # is needed for _make_op api.
        self.onnx_context = {"opset_version": 19}
        # Maps that keep track of naming uniqueness
        self._node_names: Dict[str, int] = (
            _node_names if _node_names is not None else {}
        )
        self._tensor_names: Dict[str, int] = (
            _tensor_names if _tensor_names is not None else {}
        )

        self._parent = None
        self._executor = None
        self._compiler = None
        self._device = "cpu"
        self._device_id = 0

    def __call__(self, inputs: List[str]) -> Tuple[str, ...] | str | None:
        """Method that actually builds the model given input names. Normally should
        be overriden by sub classes.

        Args:
            inputs (List[str]): input names

        Returns:
            List[str]: output names
        """
        self.inputs = inputs.copy()
        self.outputs = []

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
                input_name = self.constant(input, f"{node_name}_Input")
                input_names.append(input_name)
        # make op
        self._operators[node_name] = (op_type, attributes)
        # record topo info
        self._nodes[node_name] = (input_names, output_names)
        return output_names

    def make_tensor_np(self, data: np.ndarray, name: str):
        tensor_name = next_name(self._tensor_names, f"{self.base_name}/{name}")
        return tensor_name, data  # _make_data(data)

    def parameter(self, data: np.ndarray, name: str = "param"):
        tensor_name, tensor = self.make_tensor_np(data, name)
        self._parameters[tensor_name] = tensor
        return tensor_name

    def constant(self, data: np.ndarray, name: str = "constant"):
        tensor_name, tensor = self.make_tensor_np(data, name)
        self._const_edges[tensor_name] = tensor
        return tensor_name

    def init_cache(
        self, tensor, updated, shape: Tuple[Union[str, int], ...], dtype: DTYPE
    ):
        if tensor in self.inputs:
            self.inputs.remove(tensor)
        self._cache.append((tensor, updated, [shape, dtype, None]))

    def dynamic_tensor(self, data: Tuple[Union[str, int], ...], dtype: DTYPE):
        for item in data:
            if isinstance(item, str):
                tensor_name = next_name(self._tensor_names, f"{self.base_name}/dynamic")
                self._dynamic_tensors[tensor_name] = (data, dtype)
                return tensor_name
        return np.array(data)

    def make_submodel(self, submodel_type, *args, **kwargs):
        kwargs["_parameters"] = self._parameters
        kwargs["_const_edges"] = self._const_edges
        kwargs["_nodes"] = self._nodes
        kwargs["_operators"] = self._operators
        kwargs["_node_names"] = self._node_names
        kwargs["_tensor_names"] = self._tensor_names
        kwargs["_dynamic_tensors"] = self._dynamic_tensors
        kwargs["_cache"] = self._cache
        model_name = kwargs.pop("model_name", submodel_type.__name__)
        kwargs["model_name"] = next_name(
            self._node_names, f"{self.base_name}/{model_name}"
        )
        submodel = submodel_type(*args, **kwargs)
        submodel.parent = self
        return submodel

    def to(self, device: str, id: int = 0):
        """
        Change the device (cpu, cuda, etc) of this model.
        """
        if self._device != device or self._device_id != id:
            self._device = device
            self._device_id = id
            if self._executor is not None:
                self._executor.dispatch(
                    find_device(self._device, self._device_id), "default"
                )

    def run(
        self,
        inputs: Dict[str, np.ndarray] | None = None,
        variable_map: Dict[str, int] | None = None,
        recompile: bool = True,
    ):
        """Run model inference with InfiniTensor.
        The model will be compiled once during the first run, and will not be recompiled
        unless variable_map is given.

        Args:
            inputs ({input_name: np.ndarray}): The inputs. Can be none or empty if inputs are absent or already given.
            variable_map ({variable_name: int}): Variable map to resolve dynamic tensors. All the dynamic
                values must be resolved or the compilation will fail.
            recompile (bool): Whether the backend graph will be recompiled. Need to be true if a new variable map is provided or
                the shape of any input (including cache) is changed. True by default.

        Returns:
            [output]: list of outputs
        """

        # Run without recompile
        input_names = self.inputs.copy()
        input_names.extend(list(self._dynamic_tensors.keys()))
        output_names = self.outputs.copy()
        cache_info: Dict[str, Tuple[Any, Any]] = {}
        for cache_in, cache_out, cache_data in self._cache:
            input_names.append(cache_in)
            if cache_out not in output_names:
                output_names.append(cache_out)
            cache_shape, cache_dtype = cache_data[0], cache_data[1]
            cache_info[cache_in] = (
                cache_dtype.onnx_type(),
                resolve_variable(cache_shape, variable_map or {}),
            )

        # Build Compiler
        if self._compiler is None:
            const_edges = {
                name: _make_data(data) for name, data in self._const_edges.items()
            }
            parameters = {
                name: _make_data(data) for name, data in self._parameters.items()
            }
            edges: Dict[str, Tensor] = {}
            if inputs is not None:
                for input in self.inputs:
                    data = inputs.get(input)
                    assert data is not None, f"Input [{input}] not provided!"
                    edges[input] = _make_tensor(
                        find_onnx_type(data.dtype), list(data.shape)
                    )
            edges.update(const_edges)
            edges.update(parameters)
            dynamic_tensors = {
                name: _make_data(
                    np.array(
                        resolve_variable(d_data, variable_map or {}),
                        dtype=d_dtype.np_type(),
                    )
                )
                for name, (d_data, d_dtype) in self._dynamic_tensors.items()
            }
            edges.update(dynamic_tensors)
            edges.update(
                {
                    name: _make_tensor(cache_type, cache_shape)
                    for name, (cache_type, cache_shape) in cache_info.items()
                }
            )
            operators = {
                name: _make_operator(self.onnx_context, op_type, attributes)
                for name, (op_type, attributes) in self._operators.items()
            }
            self._compiler = _make_compiler(
                self._nodes, operators, edges, input_names, output_names
            )
            self._executor = self._compiler.compile_on(
                find_device(self._device, self._device_id), "default", []
            )
        elif recompile or self._executor is None:
            # Set input info
            if inputs is not None:
                for i, input in enumerate(self.inputs):
                    data = inputs.get(input)
                    if data is not None:
                        self._compiler.set_input_info(
                            i, find_onnx_type(data.dtype), list(data.shape)
                        )
            # Set cache info
            for name, (cache_type, cache_shape) in cache_info.items():
                self._compiler.set_input_info(
                    input_names.index(name), cache_type, cache_shape
                )
            # Copy in dynamic tensors
            if variable_map is not None:
                for name, (d_data, d_dtype) in self._dynamic_tensors.items():
                    dynamic_tensor = np.array(
                        resolve_variable(d_data, variable_map), dtype=d_dtype.np_type()
                    )
                    self._compiler.set_input(input_names.index(name), dynamic_tensor)
            
            self._executor = self._compiler.compile_on(
                find_device(self._device, self._device_id), "default", []
            )

        ## Executor should have been created at this point
        # Copy in real input
        if inputs is not None:
            for i, input_name in enumerate(self.inputs):
                input_data = inputs.get(input_name)
                if input_data is not None:
                    self._executor.set_input(i, input_data)

        # Copy in cache if data is already present
        for cache_in, cache_out, cache_data in self._cache:
            if cache_data[2] != None:
                self._executor.set_input_blob(
                    input_names.index(cache_in), cache_data[2]
                )
        # Run
        self._executor.run()
        # Update cache
        for cache_in, cache_out, cache_data in self._cache:
            cache_data[2] = self._executor.get_output_blob(
                output_names.index(cache_out)
            )

        return [
            self._executor.get_output(i) for i, output_name in enumerate(self.outputs)
        ]

    def make_onnx(
        self,
        inputs: Dict[str, Tuple[DTYPE, List[Any]]] | Dict[str, np.ndarray],
        variable_map: Dict[str, int] | None = None,
    ):
        import onnx.helper
        import onnx.numpy_helper

        input_info = []
        for input in self.inputs:
            data = inputs.get(input)
            assert data is not None, f"Input [{input}] not provided!"
            if isinstance(data, np.ndarray) or isinstance(data, np.number):
                input_info.append(
                    onnx.helper.make_value_info(
                        input,
                        onnx.helper.make_tensor_type_proto(
                            find_onnx_type(data.dtype), list(data.shape)
                        ),
                    )
                )
            else:
                input_info.append(
                    onnx.helper.make_value_info(
                        input,
                        onnx.helper.make_tensor_type_proto(
                            data[0].onnx_type(), data[1]
                        ),
                    )
                )
        output_info = [
            onnx.helper.make_empty_tensor_value_info(output) for output in self.outputs
        ]
        for cache_in, cache_out, cache_data in self._cache:
            input_info.append(
                onnx.helper.make_value_info(
                    cache_in,
                    onnx.helper.make_tensor_type_proto(
                        cache_data[1].onnx_type(),
                        resolve_variable(cache_data[0], variable_map or {}),
                    ),
                )
            )
            if cache_out not in self.outputs:
                output_info.append(onnx.helper.make_empty_tensor_value_info(cache_out))

        nodes = [
            onnx.helper.make_node(
                op,
                inputs=self._nodes[node_name][0],
                outputs=self._nodes[node_name][1],
                **attributes,
            )
            for node_name, (op, attributes) in self._operators.items()
        ]

        initializer = []
        for name, data in self._const_edges.items():
            initializer.append(onnx.numpy_helper.from_array(data, name=name))
        for name, data in self._parameters.items():
            initializer.append(onnx.numpy_helper.from_array(data, name=name))
        for name, (dynamic_tensor, tensor_type) in self._dynamic_tensors.items():
            initializer.append(
                onnx.numpy_helper.from_array(
                    np.array(
                        resolve_variable(dynamic_tensor, variable_map or {}),
                        tensor_type.np_type(),
                    ),
                    name=name,
                )
            )
        graph = onnx.helper.make_graph(
            nodes, self.base_name, input_info, output_info, initializer
        )

        return onnx.helper.make_model(graph)

    def load_params(self, data: Dict[str, np.ndarray]):
        for name in self._parameters:
            new_data = data.get(name)
            if new_data is not None:
                if self._parameters[name].shape != new_data.shape:
                    print(
                        f"Warning: Shape mismatch for {name}, expecting {self._parameters[name].shape} but get {new_data.shape}"
                    )
                if self._parameters[name].dtype != new_data.dtype:
                    print(
                        f"Warning: Type mismatch for {name}, expecting {self._parameters[name].dtype} but get {new_data.dtype}"
                    )
                self._parameters[name] = new_data
            else:
                print(f"Warning: Value for {name} is not provided for loading.")

    #############################
    # Operator APIs
    #############################
    def neg(self, X, Y="") -> str:
        return self.make_op("Neg", {}, (X,), (Y,))[0]

    def sqrt(self, X, Y="") -> str:
        return self.make_op("Sqrt", {}, (X,), (Y,))[0]

    def sigmoid(self, X, Y="") -> str:
        return self.make_op("Sigmoid", {}, (X,), (Y,))[0]

    def silu(self, X, Y="") -> str:
        return self.mul(self.sigmoid(X), X, Y)

    def add(self, A, B, C="") -> str:
        return self.make_op("Add", {}, (A, B), (C,))[0]

    def sub(self, A, B, C="") -> str:
        return self.make_op("Sub", {}, (A, B), (C,))[0]

    def mul(self, A, B, C="") -> str:
        return self.make_op("Mul", {}, (A, B), (C,))[0]

    def div(self, A, B, C="") -> str:
        return self.make_op("Div", {}, (A, B), (C,))[0]

    def pow(self, A, B, C="") -> str:
        return self.make_op("Pow", {}, (A, B), (C,))[0]

    def matmul(self, A, B, Y="") -> str:
        return self.make_op("MatMul", {}, (A, B), (Y,))[0]

    def gemm(self, A, B, C=None, Y="", alpha=1.0, beta=1.0, transA=0, transB=0) -> str:
        inputs = (A, B, C) if C is not None else (A, B)
        return self.make_op(
            "Gemm",
            {"alpha": alpha, "beta": beta, "transA": transA, "transB": transB},
            inputs,
            (Y,),
        )[0]

    def reshape(self, data, shape, reshaped="") -> str:
        return self.make_op("Reshape", {}, (data, shape), (reshaped,))[0]

    def expand(self, input, shape, output="") -> str:
        return self.make_op("Expand", {}, (input, shape), (output,))[0]

    def transpose(self, data, perm: List[int], transposed="") -> str:
        return self.make_op("Transpose", {"perm": perm}, (data,), (transposed,))[0]

    def squeeze(self, data, axes: int | List[int], squeezed="") -> str:
        return self.make_op("Squeeze", {}, (data, np.array(axes)), (squeezed,))[0]

    def unsqueeze(self, data, axes: int | List[int], unsqueezed="") -> str:
        if isinstance(axes, int):
            axes = [axes]
        return self.make_op("Unsqueeze", {}, (data, np.array(axes)), (unsqueezed,))[0]

    def gather(self, data, indices, axis=0, output="") -> str:
        return self.make_op("Gather", {"axis": axis}, (data, indices), (output,))[0]

    def slice(self, data, axis, start=0, end=9223372036854775807, step=1) -> str:
        def parse_input(input):
            if isinstance(input, int):
                return np.array([input])
            elif isinstance(input, list):
                return np.array(input)
            elif isinstance(input, str):
                return input
            else:
                raise RuntimeError("Invalid input for Slice.")

        return self.make_op(
            "Slice",
            {},
            (
                data,
                parse_input(start),
                parse_input(end),
                parse_input(axis),
                parse_input(step),
            ),
        )[0]

    def concat(self, inputs: Tuple[str, ...], axis, result="") -> str:
        return self.make_op("Concat", {"axis": axis}, inputs, (result,))[0]

    def split(self, input, axis, num_outputs) -> Tuple[str, ...]:
        outputs = self.make_op(
            "Split", {"axis": axis, "num_outputs": num_outputs}, (input,), num_outputs
        )
        return tuple(outputs)

    def reduce_sum(self, data, axes: List[int] | int, reduced="", keepdims=1) -> str:
        return self.make_op(
            "ReduceSum", {"keepdims": keepdims}, (data, np.array(axes)), (reduced,)
        )[0]

    def reduce_mean(self, data, axes: List[int] | int, reduced="", keepdims=1) -> str:
        if isinstance(axes, int):
            axes = [axes]
        return self.make_op(
            "ReduceMean", {"keepdims": keepdims}, (data, np.array(axes)), (reduced,)
        )[0]

    def cast(self, input, to: DTYPE, output="") -> str:
        return self.make_op("Cast", {"to": to.onnx_type()}, (input,), (output,))[0]

    def softmax(self, input, axis=-1, output="") -> str:
        return self.make_op("Softmax", {"axis": axis}, (input,), (output,))[0]
