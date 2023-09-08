#include "graph/graph.h"
#include "common/error_handler.h"
#include "graph/edge_info.h"
#include "infer/infer.h"

using namespace refactor::common;
namespace refactor::graph {

    Graph::Graph(graph_topo::Graph<Node, Edge> &&internal)
        : _internal(std::forward<graph_topo::Graph<Node, Edge>>(internal)) {}

    InferResult infer(Operator const &node, Edges inputs) {
        switch (node.opType.underlying()) {
            case OpType::Relu:
            case OpType::Sqrt:
                return inferUnary(node, std::move(inputs));
            case OpType::Reshape:
                return inferReshape(node, std::move(inputs));
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div:
                return inferArithmetic(node, std::move(inputs));
            case OpType::Gemm:
                return inferGemm(node, std::move(inputs));
            case OpType::MatMul:
                return inferMatMul(node, std::move(inputs));
            case OpType::CumSum:
                return inferCumSum(node, std::move(inputs));
            case OpType::Slice:
                return inferSlice(node, std::move(inputs));
            case OpType::Shape:
                return inferShape(node, std::move(inputs));
            case OpType::Where:
                return inferWhere(node, std::move(inputs));
            case OpType::Squeeze:
            case OpType::Unsqueeze:
                return inferSqueeze(node, std::move(inputs));
            case OpType::Equal:
                return inferEqual(node, std::move(inputs));
            case OpType::Softmax:
                return inferSoftmax(node, std::move(inputs));
            case OpType::Pow:
                return inferPow(node, std::move(inputs));
            case OpType::ReduceMax:
            case OpType::ReduceMin:
            case OpType::ReduceSum:
            case OpType::ReduceMean:
                return inferReduce(node, std::move(inputs));
            case OpType::Concat:
                return inferConcat(node, std::move(inputs));
            case OpType::Gather:
                return inferGather(node, std::move(inputs));
            case OpType::Cast:
                return inferCast(node, std::move(inputs));
            case OpType::Max:
                return inferMax(node, std::move(inputs));
            case OpType::Transpose:
                return inferTranspose(node, std::move(inputs));
            case OpType::Expand:
                return inferExpand(node, std::move(inputs));

            default:
                TODO("Not implemented yet");
        }
    }

    std::unordered_set<std::string> Graph::fillEdgeInfo() {
        std::unordered_set<std::string> unknownVariables;// 未知变量，将返回。
        std::unordered_set<void *> unknownEdges;         // 未知边，有入边未知的节点无法推导。
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            // 构造入边
            std::vector<Edge> inputs_(inputs.size());
            for (auto i = 0; i < inputs.size(); ++i) {
                auto ptr = (inputs_[i] = _internal.edges[inputs[i]]).get();
                if (unknownEdges.find(ptr) != unknownEdges.end()) {
                    continue;// 有入边未知，跳过节点
                }
            }
            // 推导
            auto infered = infer(_internal.nodes[nodeIdx]->operator_(), std::move(inputs_));
            if (infered.isErr()) {
                // 推导失败，记录未知变量和边
                auto error = infered.unwrapErr();
                if (std::holds_alternative<UnknownVariable>(error.value)) {
                    unknownVariables.insert(std::get<UnknownVariable>(error.value).name);
                    for (auto i = 0; i < outputs.size(); ++i) {
                        unknownEdges.insert(_internal.edges[outputs[i]].get());
                    }
                } else {
                    throw error;
                }
            } else {
                // 推导成功，填充边信息
                auto infered_ = infered.unwrap();
                if (infered_.size() < outputs.size()) {
                    OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
                } else {
                    for (auto i = 0; i < outputs.size(); ++i) {
                        _internal.edges[outputs[i]] = infered_[i];
                    }
                }
            }
        }
        return unknownVariables;
    }

}// namespace refactor::graph
