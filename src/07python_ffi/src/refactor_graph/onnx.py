from onnx import ModelProto, NodeProto, AttributeProto
from onnx.numpy_helper import to_array
from onnx.external_data_helper import ExternalDataInfo
from typing import Any
from functools import reduce
from python_ffi import (
    Compiler,
    Tensor,
    _make_data,
    _make_data_ex,
    _make_tensor,
    _make_compiler,
    _make_operator,
)


def make_compiler(model: ModelProto, external_data_path: str = "") -> Compiler:
    edges: dict[str, Tensor] = dict()
    for tensor in model.graph.initializer:
        if tensor.data_location == 1:
            edi = ExternalDataInfo(tensor)
            edges[tensor.name] = _make_data_ex(
                tensor.data_type,
                tensor.dims,
                external_data_path + "/" + edi.location,
                edi.offset,
            )
        else:
            edges[tensor.name] = _make_data(to_array(tensor))
    for tensor in model.graph.input:
        if tensor.name not in edges:
            edges[tensor.name] = _make_tensor(
                tensor.type.tensor_type.elem_type,
                [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in tensor.type.tensor_type.shape.dim
                ],
            )

    names = {}
    for node in model.graph.node:
        if node.name == "":
            node.name = "missing_name"
        if node.name in names:
            names[node.name] += 1
            node.name += "_" + str(names[node.name])
        else:
            names[node.name] = 0

    return _make_compiler(
        {node.name: (node.input, node.output) for node in model.graph.node},
        {
            node.name: _make_operator(node.op_type, _parse_attribute(node))
            for node in model.graph.node
        },
        edges,
        [i.name for i in model.graph.input],
        [o.name for o in model.graph.output],
    )


def _raise(attr: AttributeProto) -> None:
    raise NotImplementedError("Unsupported Attribute Type: {}".format(attr.type))


def _parse_attribute(node: NodeProto) -> dict[str, Any]:
    return {
        attr.name: attr.i
        if attr.type == AttributeProto.INT
        else attr.ints
        if attr.type == AttributeProto.INTS
        else attr.f
        if attr.type == AttributeProto.FLOAT
        else attr.floats
        if attr.type == AttributeProto.FLOATS
        else attr.s
        if attr.type == AttributeProto.STRING
        else attr.strings
        if attr.type == AttributeProto.STRINGS
        else _make_data(to_array(attr.t))
        if attr.type == AttributeProto.TENSOR
        else [_make_data(to_array(t)) for t in attr.tensors]
        if attr.type == AttributeProto.TENSORS
        else _raise(attr)
        for attr in node.attribute
    }
