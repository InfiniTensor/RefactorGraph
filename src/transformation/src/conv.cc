#include "transformation/conv.h"

using namespace common;
using namespace refactor::graph;

namespace refactor::transformation {
    using GraphTopo_ = GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>>;
    using Searcher_ = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>;

    ConvTransformer::ConvTransformer(graph::GraphMut &&graph)
        : _graph(std::forward<GraphMut>(graph)),
          _convs{} {
        for (auto const &node : _graph.topo().nodes()) {
            auto &info = node.info().value;
            if (!info.isSubgraph() && info.operator_().opType == OpType::Conv) {
                if (auto const &kernel = node.inputs()[1].info().value.tensor();
                    kernel.shape.size() == 4 && kernel.shape[2] == 1 && kernel.shape[3] == 1) {
                    _convs[node] = Rule::_1x1;
                } else if (auto const &dilations = info.operator_().attributes["dilations"].ints();
                           std::any_of(dilations.begin(), dilations.end(), [](auto x) { return x > 1; })) {
                    _convs[node] = Rule::Dilation;
                }
            }
        }
        std::vector<std::vector<Searcher_::Node>> subgraphs;
        for (auto const &[node, _] : _convs) {
            subgraphs.push_back({node});
        }
        _graph.extract(std::move(subgraphs));
    }

    void ConvTransformer::transform() {
    }

}// namespace refactor::transformation
