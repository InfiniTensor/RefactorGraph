#include "graph.h"
#include "infer/infer.h"
#include <algorithm>

using namespace refactor::common;

namespace refactor::graph {
    using Node = GraphTopoSearcher<NodeInfo, EdgeInfo>::Node;
    using Edge = GraphTopoSearcher<NodeInfo, EdgeInfo>::Edge;

    std::vector<EdgeInfo> takeInfo(std::vector<Edge> inputs) {
        std::vector<EdgeInfo> info(inputs.size());
        std::transform(inputs.begin(), inputs.end(), info.begin(),
                       [](Edge edge) { return std::move(edge.info()); });
        return info;
    }

    void putInfo(Node &node, std::vector<EdgeInfo> infered) {
        std::vector<Edge> outputs;
        node.outputs();
        if (infered.size() < outputs.size()) {
            throw std::runtime_error("infered.size() < outputs");
        } else {
            for (auto i = 0; i < outputs.size(); ++i) {
                outputs[i].info() = infered[i];
            }
        }
    }

    void Graph::fillEdgeInfo() {
        auto nodes = topoSearcher.nodes();

        for (auto node : nodes) {
            auto info = takeInfo(node.inputs());
            switch (node.info().opType.underlying()) {
                case OpType::Abs:
                    putInfo(node, inferAbs(info));
                    break;
                case OpType::Acos:
                case OpType::Acosh:
                    break;

                default:
                    break;
            }
        }
    }
}// namespace refactor::graph
