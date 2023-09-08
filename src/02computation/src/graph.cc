#include "graph/graph.h"
#include "common/error_handler.h"
#include "graph/edge_info.h"
#include "infer/infer.h"

using namespace refactor::common;

namespace refactor::graph {

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

    void Graph::fillEdgeInfo() {
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            std::vector<Edge> inputs_(inputs.size());
            std::transform(inputs.begin(), inputs.end(), inputs_.begin(),
                           [this](size_t idx) { return _internal.edges[idx]; });
            auto infered = infer(_internal.nodes[nodeIdx]->operator_(), std::move(inputs_));
            if (infered.isErr()) {
                throw infered.unwrapErr();
            } else {
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
    }

}// namespace refactor::graph
