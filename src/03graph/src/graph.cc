#include "graph/graph.h"
#include "common/error_handler.h"
#include "infer/infer.h"

using namespace refactor::common;

namespace refactor::graph {
    using Node = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>::Node;
    using Edge = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>::Edge;

    std::vector<EdgeInfo> takeInfo(std::vector<Edge> inputs) {
        std::vector<EdgeInfo> info(inputs.size());
        std::transform(inputs.begin(), inputs.end(), info.begin(),
                       [](Edge edge) { return std::move(edge.info().value); });
        return info;
    }

    void putInfo(Node const &node, std::vector<EdgeInfo> infered) {
        auto const outputs = node.outputs();
        if (infered.size() < outputs.size()) {
            OUT_OF_RANGE("outputs more than infered", infered.size(), outputs.size());
        } else {
            for (auto i = 0; i < outputs.size(); ++i) {
                outputs[i].info().value = infered[i];
            }
        }
    }

    GraphTopoSearcher<NodeInfo, EdgeInfo> const &Graph::topo() const {
        return searcher;
    }

    GraphMut::GraphMut(GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>> &&topo) : _topo(topo) {}

    void GraphMut::fillEdgeInfo() {
        for (auto node : _topo.nodes()) {
            auto info = takeInfo(node.inputs());
            switch (node.info().value.opType.underlying()) {
                case OpType::Abs:
                    putInfo(node, inferAbs(info));
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
                    putInfo(node, inferTrigonometry(info));
                    break;
                case OpType::Tanh:
                    putInfo(node, inferTanh(info));
                    break;
                case OpType::Add:
                case OpType::Sub:
                case OpType::Mul:
                case OpType::Div:
                    putInfo(node, inferArithmetic(info));
                    break;
                default:
                    break;
            }
        }
    }
}// namespace refactor::graph
