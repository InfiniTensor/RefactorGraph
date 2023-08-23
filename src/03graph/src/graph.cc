#include "graph/graph.h"
#include "common/error_handler.h"
#include "infer/infer.h"

using namespace refactor::common;

namespace refactor::graph {
    using Node = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>::Node;
    using Edge = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>::Edge;

    std::vector<EdgeInfo> cloneInfo(std::vector<Edge> const &inputs) {
        std::vector<EdgeInfo> info(inputs.size());
        std::transform(inputs.begin(), inputs.end(), info.begin(),
                       [](Edge edge) { return edge.info().value; });
        return info;
    }

    void putInfo(Node const &node, InferResult infered) {
        //TODO
        auto infered_ = infered.unwrap();
        auto const outputs = node.outputs();
        if (infered_.size() < outputs.size()) {
            OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
        } else {
            for (auto i = 0; i < outputs.size(); ++i) {
                outputs[i].info().value = infered_[i];
            }
        }
    }

    GraphMut::GraphMut(GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>> &&topo)
        : _topo(std::forward<GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>>>(topo)) {}

    void GraphMut::fillEdgeInfo() {
        for (auto node : _topo.nodes()) {
            auto info = cloneInfo(node.inputs());
            auto opType = node.info().value.operator_().opType;
            switch (opType.underlying()) {
                case OpType::Abs:
                case OpType::Relu:
                case OpType::PRelu:
                    putInfo(node, inferUnary(info, isNumbericDataType));
                    break;
                case OpType::Acos:
                case OpType::Acosh:
                case OpType::Asin:
                case OpType::Asinh:
                case OpType::Atan:
                case OpType::Atanh:
                case OpType::Cos:
                case OpType::Cosh:
                case OpType::Sin:
                case OpType::Sinh:
                case OpType::Tan:
                    putInfo(node, inferUnary(info, isIeee754DataType));
                    break;
                case OpType::Tanh:
                    putInfo(node, inferUnary(info, isFloatDataType));
                    break;
                case OpType::Add:
                case OpType::Sub:
                case OpType::Mul:
                case OpType::Div:
                    putInfo(node, inferArithmetic(info, opType));
                    break;
                case OpType::Gemm: {
                    auto const &attributes = node.info().value.operator_().attributes;
                    auto transA = attributes.find("transA");
                    auto transB = attributes.find("transB");
                    putInfo(node, inferGemm(info,
                                            transA == attributes.end() ? false : transA->second.int_() != 0,
                                            transB == attributes.end() ? false : transB->second.int_() != 0));
                } break;
                case OpType::Conv: {
                    auto const &attributes = node.info().value.operator_().attributes;
                    ShapeOrNot dilations = std::nullopt, pads = std::nullopt, strides = std::nullopt;
                    if (auto it = attributes.find("dilations"); it != attributes.end()) {
                        auto const &val = it->second.ints();
                        dilations = {Shape(val.begin(), val.end())};
                    }
                    if (auto it = attributes.find("pads"); it != attributes.end()) {
                        auto const &val = it->second.ints();
                        pads = Shape(val.begin(), val.end());
                    }
                    if (auto it = attributes.find("strides"); it != attributes.end()) {
                        auto const &val = it->second.ints();
                        strides = Shape(val.begin(), val.end());
                    }
                    putInfo(node, inferConv(info,
                                            std::move(dilations),
                                            std::move(pads),
                                            std::move(strides)));
                } break;
                case OpType::AveragePool:
                case OpType::MaxPool:
                case OpType::LpPool: {
                    auto const &attributes = node.info().value.operator_().attributes;
                    ShapeOrNot dilations = std::nullopt, pads = std::nullopt, strides = std::nullopt;
                    Shape kernelShape;
                    if (auto it = attributes.find("dilations"); it != attributes.end()) {
                        auto const &val = it->second.ints();
                        dilations = {Shape(val.begin(), val.end())};
                    }
                    if (auto it = attributes.find("kernel_shape"); it != attributes.end()) {
                        auto const &val = it->second.ints();
                        kernelShape = Shape(val.begin(), val.end());
                    } else {
                        RUNTIME_ERROR("Required attribute `kernel_shape` not found");
                    }
                    if (auto it = attributes.find("pads"); it != attributes.end()) {
                        auto const &val = it->second.ints();
                        pads = Shape(val.begin(), val.end());
                    }
                    if (auto it = attributes.find("strides"); it != attributes.end()) {
                        auto const &val = it->second.ints();
                        strides = Shape(val.begin(), val.end());
                    }
                    putInfo(node, inferPool(info,
                                            std::move(dilations),
                                            std::move(kernelShape),
                                            std::move(pads),
                                            std::move(strides)));
                } break;
                case OpType::GlobalAveragePool:
                case OpType::GlobalMaxPool:
                case OpType::GlobalLpPool:
                    putInfo(node, inferGlobalPool(info));
                    break;
                case OpType::Reshape:
                    putInfo(node, inferReshape(info));
                    break;
                case OpType::BatchNormalization: {
                    auto const &attributes = node.info().value.operator_().attributes;
                    auto training = false;
                    if (auto it = attributes.find("training"); it != attributes.end()) {
                        training = it->second.int_() != 0;
                    }
                    putInfo(node, inferBatchNormalization(info, training));
                } break;
                default:
                    break;
            }
        }
    }

    GraphTopo<NodeInfo, EdgeInfo> GraphMut::intoGraphTopo() {
        return _topo.intoGraphTopo().map<NodeInfo, EdgeInfo>(
            [](Cell<NodeInfo> &&node) { return std::move(node.value); },
            [](Cell<EdgeInfo> &&edge) { return std::move(edge.value); });
    }

    Graph::Graph(GraphTopo<NodeInfo, EdgeInfo> &&topo)
        : _topo(std::forward<GraphTopo<NodeInfo, EdgeInfo>>(topo)) {}

    GraphTopoSearcher<NodeInfo, EdgeInfo> const &Graph::topo() const {
        return _topo;
    }

}// namespace refactor::graph
